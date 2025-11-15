#!/usr/bin/env python
# grad_cam.py
#
# 对使用 train.py 训练好的 MobileNetV3-Small 模型做 Grad-CAM 可视化。
# - 自动从 dataset/train 读取类别名
# - 支持任意数量类别
# - 默认对预测的 top-1 类别生成 heatmap
# - 输出一张叠加热力图图片：<image>_gradcam.jpg

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms, models
from PIL import Image


# ==== 和 train.py 保持一致的预处理 ====
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD),
    ])


# ==== 和 train.py 一致的模型结构 ====
def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# ==== Grad-CAM 核心类 ====
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._forward_hook = target_layer.register_forward_hook(
            self._forward_hook_fn
        )
        self._backward_hook = target_layer.register_full_backward_hook(
            self._backward_hook_fn
        )

    def _forward_hook_fn(self, module, input, output):
        # output: [B, C, H, W]
        self.activations = output.detach()

    def _backward_hook_fn(self, module, grad_input, grad_output):
        # grad_output[0]: [B, C, H, W]
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()

    def generate(self, input_tensor: torch.Tensor, target_category: int | None = None):
        """
        input_tensor: [1, 3, H, W]
        target_category: 类别 index；None 时用模型预测的 argmax。
        返回:
            heatmap: [H, W], 值在 [0, 1]
            used_class: 最终使用的类别 index
        """
        self.model.zero_grad()

        output = self.model(input_tensor)  # [1, num_classes]

        if target_category is None:
            target_category = int(output.argmax(dim=1).item())

        target_score = output[0, target_category]
        target_score.backward()

        gradients = self.gradients      # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]

        # GAP 得到每个通道的权重 alpha_k
        alpha = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        weighted = (alpha * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]

        heatmap = weighted.squeeze(0).squeeze(0)  # [H, W]
        heatmap = torch.relu(heatmap)

        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap.cpu().numpy(), int(target_category)


# ==== 工具函数 ====
def denormalize_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """
    输入: img_tensor [3, H, W] (已经 Normalize)
    输出: BGR uint8 图像 [H, W, 3]，方便给 cv2 用
    """
    img = img_tensor.cpu().numpy()  # [3, H, W]
    img = np.transpose(img, (1, 2, 0))  # [H, W, 3]

    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std  = np.array(IMAGENET_STD).reshape(1, 1, 3)

    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def overlay_heatmap(img_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5):
    """
    img_bgr: 原图 BGR [H, W, 3]
    heatmap: [H, W] in [0, 1]
    """
    h, w = img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)
    return overlay


# ==== CLI 参数 ====
def parse_args():
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualization for the vitamin-deficiency CV model."
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image."
    )
    parser.add_argument(
        "--weights", type=str, default="best_model.pth",
        help="Path to trained model weights (.pth)."
    )
    parser.add_argument(
        "--data_root", type=str, default="dataset",
        help="Root folder containing train/val/test for class names."
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="'auto', 'cpu', or 'cuda'."
    )
    parser.add_argument(
        "--target_class", type=int, default=None,
        help="Target class index for Grad-CAM. If None, use predicted class."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output image path. Default: <image>_gradcam.jpg"
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_class_names(data_root: str):
    train_dir = Path(data_root) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    ds = datasets.ImageFolder(root=str(train_dir))
    return ds.classes


def main():
    args = parse_args()
    device = get_device(args.device)
    print("Using device:", device)

    # 1. 读取类别名，保证和 train.py 一致（ImageFolder 顺序）
    class_names = load_class_names(args.data_root)
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes:")
    for i, name in enumerate(class_names):
        print(f"  [{i}] {name}")

    # 2. 构建模型并加载权重
    model = build_model(num_classes)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. 选目标层（MobileNetV3-Small 的最后一个 feature block）
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    # 4. 读图并预处理
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    transform = get_eval_transform()
    pil_img = Image.open(str(img_path)).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)  # [1, 3, H, W]
    input_tensor.requires_grad_(True)

    # 先 forward 一次拿预测和概率
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_name = class_names[pred_idx]
    print(f"\nPredicted: [{pred_idx}] {pred_name} (prob={probs[pred_idx]:.4f})")

    # 5. 生成 Grad-CAM heatmap
    target_cls = args.target_class if args.target_class is not None else pred_idx
    print(f"Grad-CAM target class: [{target_cls}] {class_names[target_cls]}")

    # 重新跑一遍 forward + backward（上面 logits 已经用掉了，确保梯度干净）
    input_tensor.grad = None
    model.zero_grad()
    heatmap, used_class = grad_cam.generate(input_tensor, target_category=target_cls)
    grad_cam.remove_hooks()

    # 6. 反归一化 + 叠加热力图
    img_bgr = denormalize_to_uint8(input_tensor[0])
    overlay = overlay_heatmap(img_bgr, heatmap, alpha=0.5)

    # 7. 保存结果
    if args.output is None:
        out_path = img_path.with_name(img_path.stem + "_gradcam.jpg")
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"\nGrad-CAM saved to: {out_path}")


if __name__ == "__main__":
    main()
