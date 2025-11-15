# split_dataset.py
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

SOURCE_ROOT = Path("final_dataset")
TARGET_ROOT = Path("dataset")

train_ratio = 0.7
val_ratio = 0.15   # test also 0.15
test_ratio = 0.15

def main():
    TARGET_ROOT.mkdir(exist_ok=True)
    for split in ["train", "val", "test"]:
        (TARGET_ROOT / split).mkdir(exist_ok=True)

    class_dirs = [d for d in SOURCE_ROOT.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"Processing class: {class_name}")

        # all images
        img_paths = [p for p in class_dir.iterdir() if p.is_file()]

        # generate labels
        labels = [class_name] * len(img_paths)

        # train / temp (val+test)
        train_imgs, temp_imgs, _, temp_labels = train_test_split(
            img_paths,
            labels,
            test_size=(1 - train_ratio),
            stratify=labels,
            random_state=42
        )

        # val / test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=(1 - val_size),
            stratify=temp_labels,
            random_state=42
        )

        splits = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs,
        }

        for split_name, paths in splits.items():
            target_dir = TARGET_ROOT / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in paths:
                dst = target_dir / src.name
                shutil.copy2(src, dst)

    print("Done splitting!")

if __name__ == "__main__":
    main()
