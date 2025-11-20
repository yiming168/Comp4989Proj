# Run: python inference.py --image path/to/img.jpg or python inference.py --dir path/to/folder.

import argparse, json, pathlib
from typing import List
import torch
from torch import nn
from torchvision import datasets, transforms, models
from PIL import Image

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=pathlib.Path, help="Path to one image")
    parser.add_argument("--dir", type=pathlib.Path, help="Folder of images")
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

    results = [predict(model, class_names, p) for p in paths]
    for r in results:
        print(json.dumps(r, indent=2))

if __name__ == "__main__":
    main()
