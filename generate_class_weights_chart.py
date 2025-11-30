import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets


def compute_class_counts_and_weights(
    train_root: Path,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Compute per-class sample counts and class weights
    using the same formula as in train.py.
    """
    if not train_root.exists():
        raise FileNotFoundError(f"Train directory not found: {train_root}")

    ds = datasets.ImageFolder(root=str(train_root))
    class_names: List[str] = ds.classes

    targets = np.array(ds.targets, dtype=int)
    num_classes = len(class_names)
    counts = np.bincount(targets, minlength=num_classes).astype(float)

    # Same weighting strategy as train.py
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(weights)

    return class_names, counts, weights


def plot_counts_vs_weights(
    class_names: List[str],
    counts: np.ndarray,
    weights: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot a dual-axis bar chart:
    - Bar 1: number of samples per class
    - Bar 2: corresponding class weight
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indices = np.arange(len(class_names))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Sample count bars (left axis)
    bars_count = ax1.bar(
        indices - width / 2,
        counts,
        width,
        label="Sample Count",
        color="skyblue",
    )
    ax1.set_ylabel("Number of Samples")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(class_names, rotation=45, ha="right")

    # Class weight bars (right axis)
    ax2 = ax1.twinx()
    bars_weight = ax2.bar(
        indices + width / 2,
        weights,
        width,
        label="Class Weight",
        color="orange",
    )
    ax2.set_ylabel("Class Weight")

    fig.suptitle(
        "Class Sample Count vs. Loss Weight\n"
        "(minority classes receive higher weights)",
        fontsize=11,
    )

    # Combined legend
    fig.legend(
        handles=[bars_count, bars_weight],
        labels=["Sample Count", "Class Weight"],
        loc="upper right",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a chart showing class sample counts vs. "
            "the corresponding Weighted Cross-Entropy class weights."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset",
        help="Dataset root containing train/val/test (default: dataset).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/chart_class_counts_vs_weights.png",
        help=(
            "Output image path for the chart "
            "(default: figures/chart_class_counts_vs_weights.png)."
        ),
    )

    args = parser.parse_args()

    train_root = Path(args.data_root) / "train"
    output_path = Path(args.output)

    class_names, counts, weights = compute_class_counts_and_weights(train_root)
    plot_counts_vs_weights(class_names, counts, weights, output_path)

    print(f"Saved class counts vs weights chart to: {output_path}")


if __name__ == "__main__":
    main()

