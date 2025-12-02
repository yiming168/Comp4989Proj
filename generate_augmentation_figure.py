import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

"""
Usage:
python generate_augmentation_figure.py --image final_dataset/Class_02_Iron_Koilonychia/spoon_nail_10.png
"""


def generate_augmentation_panel(image_path: Path, output_path: Path) -> None:
    """
    Generate a 1x4 panel:
    Original + 3 augmented versions (rotated, darker, cropped).
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGB")

    # 1) Rotation (small angle)
    rotated = img.rotate(15, resample=Image.BILINEAR, expand=True)

    # 2) Darker (reduce brightness)
    darker = ImageEnhance.Brightness(img).enhance(0.5)

    # 3) Cropped (center crop 70% of the shorter side)
    w, h = img.size
    crop_size = int(min(w, h) * 0.7)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    cropped = img.crop((left, top, right, bottom))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    images = [img, rotated, darker, cropped]
    titles = ["Original", "Rotated", "Darker", "Cropped"]

    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(
        "Figure 3: Examples of Data Augmentation used to increase dataset diversity.",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an 'Original vs Augmented' figure: "
            "Original + rotated + darker + cropped."
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the source image (e.g., a Spoon Nail / Koilonychia example).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/figure3_data_augmentation.png",
        help=(
            "Output image path for the panel "
            "(default: figures/figure3_data_augmentation.png)."
        ),
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)

    generate_augmentation_panel(image_path, output_path)
    print(f"Saved augmentation figure to: {output_path}")


if __name__ == "__main__":
    main()

