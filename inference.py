# Run: python inference.py --image path/to/img.jpg or python inference.py --dir path/to/folder.

import argparse, json, pathlib
from typing import List
import cv2
import torch
from torch import nn
from torchvision import datasets, transforms, models
from PIL import Image
from grad_cam import GradCAM, overlay_heatmap, denormalize_to_uint8

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
    return {
        "image": str(image_path),
        "pred_class": class_names[top_idx],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=pathlib.Path, help="Path to one image")
    parser.add_argument("--dir", type=pathlib.Path, help="Folder of images")
    parser.add_argument("--grad-cam", action="store_true",
                        help="Also save a Grad-CAM overlay next to each image.")
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

    results = []
    for p in paths:
        res = predict(model, class_names, p)
        if args.grad_cam:
            target_idx = class_names.index(res["pred_class"])
            cam_path = generate_grad_cam(model, p, target_idx)
            res["grad_cam"] = str(cam_path)
            print(f"Grad-CAM saved to: {cam_path}")
        results.append(res)

    for r in results:
        print(json.dumps(r, indent=2))

if __name__ == "__main__":
    main()
