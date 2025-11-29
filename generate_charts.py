import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

"""
Usage:
generate 1-4 graphs:
python generate_charts.py --figure all --root final_dataset --history training_history.npz --data_root dataset --weights best_model.pth

generate 1:
python generate_charts.py --figure 1 --root final_dataset --output figures/figure1_class_distribution.png

generate 2:
python generate_charts.py --figure 2 --history training_history.npz
it will generate:
Loss chart → figures/figure2a_loss_curves.png
Macro-F1 chart → figures/figure2b_f1_curves.png

generate 3:
python generate_charts.py --figure 3 --data_root dataset --weights best_model.pth

generate 4:
python generate_charts.py --figure 4 --data_root dataset --weights best_model.pth
"""

# -------------------------
# Figure 1 – Class distribution
# -------------------------

def count_images_per_class(root_dir: Path) -> Tuple[List[str], List[int]]:
    """
    Count image files in each immediate subdirectory of root_dir.

    Returns:
        class_names: list of subfolder names.
        class_counts: corresponding image counts.
    """
    class_names: List[str] = []
    class_counts: List[int] = []

    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory not found or not a directory: {root_dir}")

    # Count files in each class folder (non-recursive inside each class)
    for subdir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        image_count = sum(1 for f in subdir.iterdir() if f.is_file())
        class_names.append(subdir.name)
        class_counts.append(image_count)

    if not class_names:
        raise RuntimeError(f"No class subfolders found under {root_dir}")

    return class_names, class_counts


def plot_class_distribution(
    class_names: List[str],
    class_counts: List[int],
    output_path: Path,
    healthy_keywords: Tuple[str, ...] = ("healthy", "control"),
) -> None:
    """
    Plot bar chart of class distribution and save to output_path.

    Args:
        class_names: names of classes (folder names).
        class_counts: counts per class.
        output_path: where to save the PNG.
        healthy_keywords: keywords used to highlight the healthy/control bar.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_counts, color="skyblue")

    # Try to highlight the healthy/control class if present
    healthy_index: int = -1
    lowered: Dict[str, str] = {name: name.lower() for name in class_names}
    for idx, name in enumerate(class_names):
        name_lower = lowered[name]
        if any(key in name_lower for key in healthy_keywords):
            healthy_index = idx
            break

    if healthy_index >= 0:
        bars[healthy_index].set_color("orange")

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution (Pre-Augmentation)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -------------------------
# Figure 2 – Training vs validation metrics
# -------------------------

def load_training_history(history_path: Path) -> Dict[str, np.ndarray]:
    """
    Load training history for Figure 2.

    Supports:
      - .npz with keys: train_loss, val_loss, train_f1, val_f1
      - .csv with columns: epoch, train_loss, val_loss, train_f1, val_f1
    """
    if not history_path.exists():
        raise FileNotFoundError(f"Training history file not found: {history_path}")

    history: Dict[str, np.ndarray] = {}

    suffix = history_path.suffix.lower()
    if suffix == ".npz":
        data = np.load(history_path)
        for key in ("train_loss", "val_loss", "train_f1", "val_f1"):
            if key not in data:
                raise KeyError(f"Missing key '{key}' in {history_path}")
            history[key] = np.asarray(data[key], dtype=float)
    elif suffix in (".csv", ".txt"):
        with history_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise RuntimeError(f"No rows found in CSV history file: {history_path}")

        for key in ("train_loss", "val_loss", "train_f1", "val_f1"):
            try:
                history[key] = np.array([float(r[key]) for r in rows], dtype=float)
            except KeyError as exc:
                raise KeyError(f"Missing column '{key}' in CSV: {history_path}") from exc
    else:
        raise ValueError(f"Unsupported history file format: {history_path}")

    return history


def plot_training_history(
    history: Dict[str, np.ndarray],
    loss_output_path: Path,
    f1_output_path: Path,
) -> None:
    """
    Plot training vs validation loss and macro-F1 over epochs.
    """
    loss_output_path.parent.mkdir(parents=True, exist_ok=True)
    f1_output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Loss curves (Figure 2A)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_output_path, dpi=300)
    plt.close()

    # Macro-F1 curves (Figure 2B)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_f1"], label="Train Macro-F1")
    plt.plot(epochs, history["val_f1"], label="Val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training vs Validation Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f1_output_path, dpi=300)
    plt.close()


# -------------------------
# Figures 3 & 4 – Confusion matrix + per-class F1
# -------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
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


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_class_names(data_root: Path) -> List[str]:
    train_dir = data_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train dir not found for class names: {train_dir}")
    ds = datasets.ImageFolder(root=str(train_dir))
    return ds.classes


def compute_test_predictions(
    data_root: Path,
    weights_path: Path,
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Run the trained model on the test set and return (y_true, y_pred, class_names).
    """
    class_names = load_class_names(data_root)
    num_classes = len(class_names)

    model = build_model(num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    test_dir = data_root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    test_ds = datasets.ImageFolder(root=str(test_dir), transform=get_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_true: List[int] = []
    all_pred: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    return np.array(all_true), np.array(all_pred), class_names


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """
    Generate and save confusion matrix heatmap (Figure 3).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """
    Generate and save per-class F1 score bar chart (Figure 4).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores = f1_score(y_true, y_pred, average=None)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=class_names, palette="viridis")
    plt.title("Per-Class F1 Score (Test Set)")
    plt.xlabel("F1 Score")
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate result figures (Figures 1–4) for the vitamin deficiency project."
        )
    )
    parser.add_argument(
        "--figure",
        type=str,
        choices=["1", "2", "3", "4", "all"],
        default="1",
        help="Which figure to generate: 1, 2, 3, 4, or 'all' (default: 1).",
    )

    # Figure 1 (class distribution)
    parser.add_argument(
        "--root",
        type=str,
        default="final_dataset",
        help="Root directory containing one subfolder per class (Figure 1, default: final_dataset).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/figure1_class_distribution.png",
        help="Output image path for Figure 1 (default: figures/figure1_class_distribution.png).",
    )

    # Figure 2 (training history)
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Training history file (.npz or .csv) for Figure 2.",
    )
    parser.add_argument(
        "--loss_fig",
        type=str,
        default="figures/figure2a_loss_curves.png",
        help="Output image path for loss curves (Figure 2A).",
    )
    parser.add_argument(
        "--f1_fig",
        type=str,
        default="figures/figure2b_f1_curves.png",
        help="Output image path for Macro-F1 curves (Figure 2B).",
    )

    # Figures 3 & 4 (test metrics)
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset",
        help="Dataset root containing train/val/test for Figures 3 & 4 (default: dataset).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best_model.pth",
        help="Trained model weights (.pth) for Figures 3 & 4 (default: best_model.pth).",
    )
    parser.add_argument(
        "--cm_fig",
        type=str,
        default="figures/figure3_confusion_matrix.png",
        help="Output image path for confusion matrix (Figure 3).",
    )
    parser.add_argument(
        "--per_class_f1_fig",
        type=str,
        default="figures/figure4_per_class_f1.png",
        help="Output image path for per-class F1 scores (Figure 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for Figures 3 & 4: 'auto', 'cpu', or 'cuda' (default: auto).",
    )

    args = parser.parse_args()

    # Figure 1
    if args.figure in ("1", "all"):
        root_dir = Path(args.root)
        output_path = Path(args.output)
        class_names, class_counts = count_images_per_class(root_dir)
        plot_class_distribution(class_names, class_counts, output_path)
        print(f"[Figure 1] Saved class distribution chart to: {output_path}")

    # Figure 2
    if args.figure in ("2", "all"):
        if args.history is None:
            print("[Figure 2] Skipped: --history file not provided.")
        else:
            history_path = Path(args.history)
            try:
                history = load_training_history(history_path)
            except FileNotFoundError as exc:
                print(f"[Figure 2] Skipped: {exc}")
            else:
                loss_output = Path(args.loss_fig)
                f1_output = Path(args.f1_fig)
                plot_training_history(history, loss_output, f1_output)
                print(f"[Figure 2] Saved loss curves to: {loss_output}")
                print(f"[Figure 2] Saved Macro-F1 curves to: {f1_output}")

    # Figures 3 & 4
    if args.figure in ("3", "4", "all"):
        device = get_device(args.device)
        data_root = Path(args.data_root)
        weights_path = Path(args.weights)

        y_true, y_pred, class_names = compute_test_predictions(
            data_root=data_root,
            weights_path=weights_path,
            device=device,
        )

        if args.figure in ("3", "all"):
            cm_output = Path(args.cm_fig)
            plot_confusion_matrix(y_true, y_pred, class_names, cm_output)
            print(f"[Figure 3] Saved confusion matrix to: {cm_output}")

        if args.figure in ("4", "all"):
            f1_output = Path(args.per_class_f1_fig)
            plot_per_class_f1(y_true, y_pred, class_names, f1_output)
            print(f"[Figure 4] Saved per-class F1 chart to: {f1_output}")


if __name__ == "__main__":
    main()
