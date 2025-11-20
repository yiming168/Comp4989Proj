# Run: python inference.py --image path/to/img.jpg or python inference.py --dir path/to/folder.
# Add --suggestions flag to get AI-generated food recommendations.

import argparse, json, pathlib
import cv2
from typing import List, Optional
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

import torch
from torch import nn
from torchvision import datasets, transforms, models
from PIL import Image
from grad_cam import GradCAM, overlay_heatmap, denormalize_to_uint8

try:
    from food_suggestions import get_food_suggestions, format_suggestions_output, initialize_ai_client
    FOOD_SUGGESTIONS_AVAILABLE = True
except ImportError:
    FOOD_SUGGESTIONS_AVAILABLE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "dataset/train"  # only to read class names
WEIGHTS = "best_model.pth"

eval_tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def build_model(num_classes: int) -> nn.Module:
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m

def load_classes() -> List[str]:
    ds = datasets.ImageFolder(DATA_ROOT)
    return ds.classes

def load_model(class_names: List[str]) -> nn.Module:
    model = build_model(len(class_names))
    state = torch.load(WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

def predict(model: nn.Module, class_names: List[str], image_path: pathlib.Path):
    img = Image.open(image_path).convert("RGB")
    x = eval_tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    top_idx = int(torch.argmax(probs).item())
    confidence = float(probs[top_idx].item())
    return {
        "image": str(image_path),
        "pred_class": class_names[top_idx],
        "confidence": confidence,
        "probs": {cls: float(probs[i]) for i, cls in enumerate(class_names)}
    }

def generate_grad_cam(model: nn.Module, image_path: pathlib.Path, target_idx: int) -> pathlib.Path:
    """
    Generate and save a Grad-CAM overlay for the given image targeting target_idx.
    """
    img = Image.open(image_path).convert("RGB")
    x = eval_tfm(img).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    heatmap, _ = grad_cam.generate(x, target_category=target_idx)
    grad_cam.remove_hooks()

    base_img = denormalize_to_uint8(x[0])
    overlay = overlay_heatmap(base_img, heatmap, alpha=0.5)

    out_path = image_path.with_name(image_path.stem + "_gradcam.jpg")
    cv2.imwrite(str(out_path), overlay)
    return out_path

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on nutrition classification model with optional AI food suggestions"
    )
    parser.add_argument("--image", type=pathlib.Path, help="Path to one image")
    parser.add_argument("--dir", type=pathlib.Path, help="Folder of images")
    parser.add_argument("--grad-cam", action="store_true",
                        help="Also save a Grad-CAM overlay next to each image.")
    parser.add_argument(
        "--suggestions",
        action="store_true",
        help="Get AI-generated food suggestions based on prediction (requires GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output only JSON (no formatted suggestions text)"
    )
    args = parser.parse_args()

    paths = []
    if args.image:
        paths.append(args.image)
    if args.dir:
        paths.extend([p for p in args.dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not paths:
        raise SystemExit("Provide --image or --dir")

    class_names = load_classes()
    model = load_model(class_names)

    # Initialize AI model if suggestions are requested
    ai_model = None
    if args.suggestions:
        if not FOOD_SUGGESTIONS_AVAILABLE:
            raise SystemExit(
                "Food suggestions not available. Install google-generativeai: pip install google-generativeai"
            )
        try:
            ai_model = initialize_ai_client(args.api_key)
        except ValueError as e:
            raise SystemExit(f"Failed to initialize AI client: {e}")

    results = []
    for path in paths:
        result = predict(model, class_names, path)
        
        # Generate Grad-CAM if requested
        if args.grad_cam:
            target_idx = class_names.index(result["pred_class"])
            cam_path = generate_grad_cam(model, path, target_idx)
            result["grad_cam"] = str(cam_path)
            print(f"Grad-CAM saved to: {cam_path}")
        
        # Add food suggestions if requested
        if args.suggestions and ai_model:
            try:
                suggestions_result = get_food_suggestions(
                    result["pred_class"],
                    result["confidence"],
                    api_key=args.api_key,
                    model=ai_model
                )
                result["food_suggestions"] = suggestions_result["suggestions"]
                result["deficiency_info"] = suggestions_result["deficiency_info"]
            except Exception as e:
                result["food_suggestions_error"] = str(e)
        
        results.append(result)
    
    # Output results
    if args.json_only or not args.suggestions:
        # JSON output only
        print(json.dumps(results, indent=2))
    else:
        # Formatted output with suggestions
        for r in results:
            print(json.dumps({
                "image": r["image"],
                "pred_class": r["pred_class"],
                "confidence": r["confidence"],
                "probs": r["probs"]
            }, indent=2))
            
            if "food_suggestions" in r:
                print("\n" + format_suggestions_output({
                    "suggestions": r["food_suggestions"],
                    "deficiency_info": r["deficiency_info"]
                }))
            elif "food_suggestions_error" in r:
                print(f"\nError getting food suggestions: {r['food_suggestions_error']}")
            print("\n")

if __name__ == "__main__":
    main()
