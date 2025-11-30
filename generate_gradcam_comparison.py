import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from grad_cam import (
    GradCAM,
    build_model,
    denormalize_to_uint8,
    get_device,
    get_eval_transform,
    load_class_names,
    overlay_heatmap,
)


def generate_gradcam_comparison(
    image_path: Path,
    data_root: Path,
    weights_path: Path,
    output_path: Path,
    device_str: str = "auto",
    target_class: int | None = None,
) -> None:
    """
    Create a side-by-side comparison:
    Left: original image
    Right: Grad-CAM heatmap overlay for the predicted (or specified) class.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    device = get_device(device_str)
    class_names = load_class_names(str(data_root))
    num_classes = len(class_names)

    model = build_model(num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = get_eval_transform()
    pil_img = Image.open(str(image_path)).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    if target_class is None:
        target_class = pred_idx

    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    heatmap, _ = grad_cam.generate(input_tensor, target_category=target_class)
    grad_cam.remove_hooks()

    # Convert tensors to displayable images (RGB for matplotlib)
    orig_bgr = denormalize_to_uint8(input_tensor[0])
    overlay_bgr = overlay_heatmap(orig_bgr, heatmap, alpha=0.5)
    orig_rgb = orig_bgr[..., ::-1]
    overlay_rgb = overlay_bgr[..., ::-1]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=10)
    axes[1].axis("off")

    title = "Original vs Grad-CAM Heatmap"
    if 0 <= target_class < len(class_names):
        title += f" ({class_names[target_class]})"
    fig.suptitle(title, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a side-by-side comparison image: "
            "Original vs Grad-CAM heatmap overlay."
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image (e.g., a Spoon Nail / Koilonychia example).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset",
        help="Dataset root containing train/val/test (used for class names).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best_model.pth",
        help="Path to trained model weights (.pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/gradcam_comparison.png",
        help="Output image path for the comparison figure.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', or 'cuda' (default: auto).",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="Optional target class index for Grad-CAM. Default: model prediction.",
    )

    args = parser.parse_args()

    generate_gradcam_comparison(
        image_path=Path(args.image),
        data_root=Path(args.data_root),
        weights_path=Path(args.weights),
        output_path=Path(args.output),
        device_str=args.device,
        target_class=args.target_class,
    )
    print(f"Saved Grad-CAM comparison figure to: {args.output}")


if __name__ == "__main__":
    main()

