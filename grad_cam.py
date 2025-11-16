#!/usr/bin/env python
# grad_cam.py

import argparse
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import datasets, models, transforms


IMAGENET_MEAN: Sequence[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: Sequence[float] = [0.229, 0.224, 0.225]


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._forward_hook = target_layer.register_forward_hook(self._forward_hook_fn)
        self._backward_hook = target_layer.register_full_backward_hook(
            self._backward_hook_fn
        )

    def _forward_hook_fn(
        self, module: nn.Module, inputs, output: torch.Tensor
    ) -> None:
        self.activations = output.detach()

    def _backward_hook_fn(
        self, module: nn.Module, grad_input, grad_output
    ) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._forward_hook.remove()
        self._backward_hook.remove()

    def generate(
        self, input_tensor: torch.Tensor, target_category: Optional[int] = None
    ) -> tuple[np.ndarray, int]:
        """
        input_tensor: [1, 3, H, W]
        Returns (heatmap[H, W] in [0,1], used_class_index).
        """
        self.model.zero_grad()

        output = self.model(input_tensor)  # [1, num_classes]

        if target_category is None:
            target_category = int(output.argmax(dim=1).item())

        target_score = output[0, target_category]
        target_score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured for Grad-CAM.")

        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]

        alpha = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        weighted = (alpha * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]

        heatmap = weighted.squeeze(0).squeeze(0)  # [H, W]
        heatmap = torch.relu(heatmap)

        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap.cpu().numpy(), int(target_category)


def denormalize_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor [3, H, W] back to BGR uint8 [H, W, 3].
    """
    img = img_tensor.detach().cpu().numpy()  # [3, H, W]
    img = np.transpose(img, (1, 2, 0))  # [H, W, 3]

    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)

    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def overlay_heatmap(
    img_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay a heatmap [H, W] onto an image [H, W, 3] (BGR).
    """
    h, w = img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)
    return overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualization for the vitamin-deficiency CV model."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best_model.pth",
        help="Path to trained model weights (.pth).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset",
        help="Root folder containing train/val/test for class names.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="Target class index for Grad-CAM. If None, use predicted class.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Default: <image>_gradcam.jpg",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_class_names(data_root: str) -> list[str]:
    train_dir = Path(data_root) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    ds = datasets.ImageFolder(root=str(train_dir))
    return ds.classes


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print("Using device:", device)

    class_names = load_class_names(args.data_root)
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes:")
    for idx, name in enumerate(class_names):
        print(f"  [{idx}] {name}")

    model = build_model(num_classes)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    transform = get_eval_transform()
    pil_img = Image.open(str(img_path)).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)  # [1, 3, H, W]
    input_tensor.requires_grad_(True)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_name = class_names[pred_idx]
    print(f"\nPredicted: [{pred_idx}] {pred_name} (prob={probs[pred_idx]:.4f})")

    target_cls = args.target_class if args.target_class is not None else pred_idx
    print(f"Grad-CAM target class: [{target_cls}] {class_names[target_cls]}")

    input_tensor.grad = None
    model.zero_grad()
    heatmap, used_class = grad_cam.generate(input_tensor, target_category=target_cls)
    grad_cam.remove_hooks()

    img_bgr = denormalize_to_uint8(input_tensor[0])
    overlay = overlay_heatmap(img_bgr, heatmap, alpha=0.5)

    if args.output is None:
        out_path = img_path.with_name(img_path.stem + "_gradcam.jpg")
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"\nGrad-CAM saved to: {out_path}")


if __name__ == "__main__":
    main()

