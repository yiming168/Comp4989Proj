"""
Run ONNX inference on Jetson (or CPU) with optional GPU/TensorRT providers.

Usage:
  python jetson_deploy/onnx_inference.py --image path/to/img.jpg --model jetson_deploy/model.onnx
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import datasets, transforms

IMAGENET_MEAN: Sequence[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: Sequence[float] = [0.229, 0.224, 0.225]


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_classes(classes_path: Path | None, fallback_train_root: Path) -> List[str]:
    if classes_path and classes_path.exists():
        return json.loads(classes_path.read_text())
    ds = datasets.ImageFolder(root=str(fallback_train_root))
    return ds.classes


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def select_providers() -> list[str]:
    preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    return [p for p in preferred if p in available]


def infer(model_path: Path, image_path: Path, classes: List[str], tfm: transforms.Compose) -> dict:
    providers = select_providers()
    sess = ort.InferenceSession(str(model_path), providers=providers)

    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).numpy()

    logits = sess.run(["logits"], {"input": x})[0]
    probs = softmax(logits)[0]
    top_idx = int(np.argmax(probs))

    return {
        "image": str(image_path),
        "pred_class": classes[top_idx],
        "probs": {cls: float(probs[i]) for i, cls in enumerate(classes)},
        "providers": providers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX inference for Jetson.")
    parser.add_argument("--image", type=Path, required=True, help="Image to classify.")
    parser.add_argument("--model", type=Path, default=Path("jetson_deploy/model.onnx"), help="Path to ONNX model.")
    parser.add_argument("--classes", type=Path, default=Path("jetson_deploy/classes.json"), help="Optional class list JSON.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/train"), help="Fallback train root to read class names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    classes = load_classes(args.classes, args.data_root)
    result = infer(args.model, args.image, classes, get_transform())

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
