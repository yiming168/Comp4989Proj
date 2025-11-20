"""
Test script to demonstrate food_suggestions.py functionality
This shows the prompt generation without requiring an API key.
"""

from food_suggestions import (
    get_deficiency_info,
    create_food_suggestion_prompt,
    DEFICIENCY_MAP
)

def test_prompt_generation():
    """Test that prompts are generated correctly for each class."""
    
    print("Testing Food Suggestions Prompt Generation\n")
    print("=" * 60)
    
    test_cases = [
        ("Class_01_Vitamin_A_Bitots_Spot", 0.85),
        ("Class_02_Iron_Koilonychia", 0.92),
        ("Class_03_B_Vitamin_Glossitis", 0.78),
        ("Class_04_Healthy_Control", 0.95),
    ]
    
    for pred_class, confidence in test_cases:
        print(f"\n{'='*60}")
        print(f"Test Case: {pred_class}")
        print(f"Confidence: {confidence:.1%}")
        print(f"{'='*60}\n")
        
        # Get deficiency info
        deficiency_info = get_deficiency_info(pred_class)
        print("Deficiency Info:")
        print(f"  Deficiency: {deficiency_info['deficiency']}")
        print(f"  Symptom: {deficiency_info['symptom']}")
        print(f"  Description: {deficiency_info['description']}\n")
        
        # Generate prompt
        prompt = create_food_suggestion_prompt(deficiency_info, confidence)
        print("Generated Prompt:")
        print("-" * 60)
        print(prompt)
        print("-" * 60)
        print()
    
    print("\n" + "=" * 60)
    print("All test cases completed!")
    print("=" * 60)
    print("\nTo use with actual AI API:")
    print("1. Get API key from: https://ai.google.dev/aistudio/")
    print("2. Set environment variable: set GEMINI_API_KEY=your-key-here")
    print("3. Run: python food_suggestions.py Class_01_Vitamin_A_Bitots_Spot 0.85")
    print("   OR: python inference.py --image test/image.png --suggestions")

if __name__ == "__main__":
    test_prompt_generation()

