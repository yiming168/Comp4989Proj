"""
Export trained PyTorch model to ONNX for Jetson deployment.

Usage:
  python jetson_deploy/export_onnx.py --weights best_model.pth --out jetson_deploy/model.onnx
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch import nn
from torchvision import datasets, models


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_class_names(train_root: Path) -> List[str]:
    ds = datasets.ImageFolder(root=str(train_root))
    return ds.classes


def export(classes: List[str], weights_path: Path, out_path: Path) -> None:
    model = build_model(len(classes))
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX to: {out_path}")

    classes_path = out_path.with_name("classes.json")
    classes_path.write_text(json.dumps(classes, indent=2))
    print(f"Saved class list to: {classes_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to ONNX for Jetson.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/train"), help="Path to train split to read class names.")
    parser.add_argument("--weights", type=Path, default=Path("best_model.pth"), help="Trained PyTorch weights.")
    parser.add_argument("--out", type=Path, default=Path("jetson_deploy/model.onnx"), help="Output ONNX path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Train data root not found: {args.data_root}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    classes = load_class_names(args.data_root)
    print(f"Detected classes ({len(classes)}): {classes}")
    export(classes, args.weights, args.out)


if __name__ == "__main__":
    main()
