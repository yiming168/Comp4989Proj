"""
Food Suggestions API Integration
Uses AI Studio (Google Gemini API) to generate food suggestions based on nutrition deficiency predictions.
"""

import os
import json
from typing import Dict, Optional, List
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, will use environment variables only
    pass

import google.generativeai as genai


# Map model class names to deficiency types and descriptions
DEFICIENCY_MAP = {
    "Class_00_Unrelated_Control": {
        "deficiency": None,
        "symptom": "No deficiency detected",
        "description": "No nutritional deficiency detected - unrelated/healthy condition"
    },
    "Class_01_Vitamin_A_Bitots_Spot": {
        "deficiency": "Vitamin A",
        "symptom": "Bitot's Spots (eye condition)",
        "description": "Vitamin A deficiency indicated by Bitot's spots on the conjunctiva"
    },
    "Class_02_Iron_Koilonychia": {
        "deficiency": "Iron",
        "symptom": "Koilonychia (spoon-shaped nails)",
        "description": "Iron deficiency indicated by koilonychia (spoon-shaped nails)"
    },
    "Class_03_Vitamin_B_Glossitis": {
        "deficiency": "B Vitamins",
        "symptom": "Glossitis (tongue inflammation)",
        "description": "B vitamin deficiency indicated by glossitis (tongue inflammation)"
    },
    "Class_04_Vitamin_B_Angular_Cheilitis": {
        "deficiency": "B Vitamins",
        "symptom": "Angular Cheilitis (mouth corner cracks)",
        "description": "B vitamin deficiency indicated by angular cheilitis (cracks at mouth corners)"
    },
    "Class_05_Vitamin_C_Gums": {
        "deficiency": "Vitamin C",
        "symptom": "Gum problems (bleeding/swollen gums)",
        "description": "Vitamin C deficiency indicated by gum problems (bleeding, swelling, or inflammation)"
    },
    "Class_06_Vitamin_A_Keratosis_Pilaris": {
        "deficiency": "Vitamin A",
        "symptom": "Keratosis Pilaris (bumpy skin)",
        "description": "Vitamin A deficiency indicated by keratosis pilaris (bumpy, rough skin patches)"
    },
    "Class_07_Vitamin_B3_Pellagra": {
        "deficiency": "Vitamin B3 (Niacin)",
        "symptom": "Pellagra (skin rash/dermatitis)",
        "description": "Vitamin B3 (niacin) deficiency indicated by pellagra (dermatitis, diarrhea, dementia symptoms)"
    },
    "Class_08_Zinc_Deficiency_Acrodermatitis": {
        "deficiency": "Zinc",
        "symptom": "Acrodermatitis (skin inflammation)",
        "description": "Zinc deficiency indicated by acrodermatitis (skin inflammation, especially on extremities)"
    }
}


def get_deficiency_info(pred_class: str) -> Dict[str, Optional[str]]:
    """Get deficiency information from predicted class name."""
    return DEFICIENCY_MAP.get(pred_class, {
        "deficiency": None,
        "symptom": "Unknown",
        "description": "Unknown condition"
    })


def create_food_suggestion_prompt(
    deficiency_info: Dict[str, Optional[str]],
    confidence: float
) -> str:
    """Create a prompt for the AI API based on deficiency information."""
    
    if deficiency_info["deficiency"] is None:
        # Healthy case - suggest general nutrition tips
        prompt = """You are a nutrition advisor. The person appears to have no detected nutritional deficiencies. 
Please provide 5-7 general healthy eating tips and food recommendations to maintain good nutrition and prevent deficiencies.
Keep the response concise and practical."""
    else:
        # Deficiency case - suggest specific foods
        deficiency = deficiency_info["deficiency"]
        symptom = deficiency_info["symptom"]
        description = deficiency_info["description"]
        
        prompt = f"""You are a nutrition advisor. A visual screening tool has detected a potential {deficiency} deficiency 
(indicated by {symptom} - {description}). The detection confidence is {confidence:.1%}.

IMPORTANT DISCLAIMER: This is not a medical diagnosis. The person should consult a healthcare professional.

Please provide:
1. 5-7 specific food recommendations rich in {deficiency} to help address this potential deficiency
2. Brief explanation of why each food is beneficial
3. General dietary tips for improving {deficiency} intake

Keep the response concise, practical, and easy to understand. Format as a clear list."""

    return prompt


def initialize_ai_client(api_key: Optional[str] = None) -> genai.GenerativeModel:
    """
    Initialize the Google Gemini AI client.
    
    Args:
        api_key: API key for Google AI Studio. If None, reads from GEMINI_API_KEY env var.
    
    Returns:
        Initialized GenerativeModel instance.
    
    Raises:
        ValueError: If API key is not provided.
    """
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "API key not found. Please set GEMINI_API_KEY in .env file or environment variable "
            "or pass api_key parameter. Get your key from: https://ai.google.dev/aistudio/"
        )
    
    genai.configure(api_key=api_key)
    
    # Use Gemini 2.5 Flash (fast and efficient) or fallback to latest versions
    # Model names need to match exactly what's available in the API
    model_names = [
        'gemini-2.5-flash',      # Fast and efficient
        'gemini-2.5-pro',        # More capable
        'gemini-flash-latest',   # Latest flash
        'gemini-pro-latest',     # Latest pro
    ]
    
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            return model
        except Exception as e:
            if model_name == model_names[-1]:
                # Last model failed, raise error
                raise ValueError(
                    f"Failed to initialize any Gemini model. Tried: {model_names}. "
                    f"Last error: {str(e)}. Please check your API key and available models."
                )
            continue


def get_food_suggestions(
    pred_class: str,
    confidence: float,
    api_key: Optional[str] = None,
    model: Optional[genai.GenerativeModel] = None
) -> Dict[str, str]:
    """
    Get food suggestions from AI API based on classification result.
    
    Args:
        pred_class: Predicted class name from the model
        confidence: Confidence score (0.0 to 1.0)
        api_key: Optional API key (if not provided, uses env var)
        model: Optional pre-initialized model (if not provided, initializes new one)
    
    Returns:
        Dictionary with 'suggestions' (AI response) and 'deficiency_info' (metadata)
    
    Raises:
        ValueError: If API key is missing
        Exception: If API call fails
    """
    deficiency_info = get_deficiency_info(pred_class)
    prompt = create_food_suggestion_prompt(deficiency_info, confidence)
    
    # Initialize model if not provided
    if model is None:
        model = initialize_ai_client(api_key)
    
    try:
        response = model.generate_content(prompt)
        suggestions_text = response.text
        
        return {
            "suggestions": suggestions_text,
            "deficiency_info": deficiency_info,
            "prompt_used": prompt
        }
    except Exception as e:
        raise Exception(f"Failed to get food suggestions from AI API: {str(e)}")


def format_suggestions_output(result: Dict[str, str]) -> str:
    """Format the suggestions output for display."""
    deficiency_info = result["deficiency_info"]
    
    output = []
    output.append("=" * 60)
    output.append("FOOD SUGGESTIONS")
    output.append("=" * 60)
    
    if deficiency_info["deficiency"]:
        output.append(f"\nDetected Deficiency: {deficiency_info['deficiency']}")
        output.append(f"Symptom: {deficiency_info['symptom']}")
    else:
        output.append("\nStatus: No deficiency detected (Healthy)")
    
    output.append("\n" + "-" * 60)
    output.append("Food Recommendations:")
    output.append("-" * 60)
    output.append(result["suggestions"])
    output.append("\n" + "=" * 60)
    output.append("DISCLAIMER: This is not a medical diagnosis.")
    output.append("Please consult a healthcare professional for medical advice.")
    output.append("=" * 60)
    
    return "\n".join(output)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python food_suggestions.py <predicted_class> [confidence]")
        print("\nExample:")
        print("  python food_suggestions.py Class_01_Vitamin_A_Bitots_Spot 0.85")
        sys.exit(1)
    
    pred_class = sys.argv[1]
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    
    try:
        result = get_food_suggestions(pred_class, confidence)
        print(format_suggestions_output(result))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

