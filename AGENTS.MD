AGENTS.MD

Guidelines for AI coding assistants (e.g., Copilot, Codex, ChatGPT) working in this repo.

1. Project Overview

Title: Visual Screening of Vitamin Deficiency–Related Symptoms
Goal:
Train a CNN classifier that detects specific, well-defined visual symptoms of nutrient deficiencies from images:

Vitamin A – Bitot’s Spots (eye, conjunctiva)

Iron – Koilonychia (spoon nails)

B Vitamins – Glossitis (tongue)

Healthy Controls – eyes / nails / tongues without these signs

This is not a medical diagnostic tool. It is a proof-of-concept CV project for academic use.

2. Data Layout

Raw curated data from the web is stored in:

/final_dataset/
  ├── Class_01_Vitamin_A_Bitots_Spot/
  ├── Class_02_Iron_Koilonychia/
  ├── Class_03_B_Vitamin_Glossitis/
  └── Class_04_Healthy_Control/


Training code uses a train/val/test split under:

/dataset/
  ├── train/
  │   ├── Class_01_Vitamin_A_Bitots_Spot/
  │   ├── Class_02_Iron_Koilonychia/
  │   ├── Class_03_B_Vitamin_Glossitis/
  │   └── Class_04_Healthy_Control/
  ├── val/
  │   └── same class folders...
  └── test/
      └── same class folders...

IMPORTANT RULES FOR AGENTS

Never modify or delete anything in /final_dataset/.

Treat it as read-only, human-curated ground truth.

All automatic splitting / augmentation must operate on copies in /dataset/.

Assume ~100 images per symptom class + ~100 healthy as a target scale, but code should not hard-code counts.

3. Key Scripts (Current / Planned)
3.1 split_dataset.py

Purpose:

Read images from /final_dataset/

Create stratified train/val/test splits under /dataset/

Behavior:

Uses the folder names as labels (ImageFolder-compatible)

Default ratios: train 0.7, val 0.15, test 0.15

Copies files (not moves) from final_dataset → dataset

If you are editing or generating this file as an agent:

Use sklearn.model_selection.train_test_split with stratify=labels

Make split ratios configurable via constants or CLI flags

Log class counts per split to STDOUT

Sample run:

python split_dataset.py

3.2 train.py

Purpose:

Train a multi-class classifier (4 classes) using PyTorch + pretrained backbone (e.g., MobileNetV3 / ResNet18).

Requirements (for agents):

Framework: PyTorch + torchvision

Data source: torchvision.datasets.ImageFolder("dataset/train", ...)

Input size: 224 × 224 (resize & crop)

Normalization: Use ImageNet mean/std

Loss: Weighted Cross Entropy (class-balanced loss)

Compute weights from class frequencies in training set

Metrics:

Accuracy

Macro-averaged F1 (important because class imbalance)

Data augmentation for training:

RandomResizedCrop

RandomHorizontalFlip

Small RandomRotation

Some ColorJitter

Save best model by val macro F1 → best_model.pth

Expected usage:

python train.py


Hyperparameters (as constants or args):

NUM_CLASSES = 4

BATCH_SIZE ≈ 16

NUM_EPOCHS ≈ 10–30

LR ≈ 1e-4 (Adam)

3.3 inference.py (planned)

Purpose:

Load best_model.pth

Run prediction on:

Single image path, or

A folder of images

Requirements:

Output class label + probabilities (softmax)

Print mapping between class index ↔ class folder name

Simple CLI interface, e.g.:

python inference.py --image path/to/image.jpg


or

python inference.py --dir path/to/folder

3.4 grad_cam.py (planned)

Purpose:

Visualize model attention on a given input image using Grad-CAM or similar method.

Requirements:

Work with the same backbone as in train.py

Generate overlay heatmaps for:

Bitot’s Spots (eyes)

Koilonychia (nails)

Glossitis (tongue)

Save result as *_gradcam.jpg in an output folder

This script is used for model interpretability, to check if the network is looking at the correct anatomical region.

3.5 jetson_deploy/ (planned)

Create a subfolder for later Jetson deployment:

/jetson_deploy/
  ├── export_onnx.py
  ├── onnx_inference.py
  └── README.md


Agents should:

Provide export_onnx.py to convert best_model.pth → model.onnx

Provide a minimalist onnx_inference.py for Jetson inference

Keep dependencies light (no Jupyter notebooks inside this folder)

4. Environment & Dependencies

Target: Python 3.10+

Main libraries:

torch

torchvision

scikit-learn

numpy

matplotlib (optional, for Grad-CAM visualization)

opencv-python (optional, nice to have for image IO)

Suggested requirements.txt (agents can extend, but avoid bloat):

torch
torchvision
scikit-learn
numpy
matplotlib
opencv-python

5. Coding Conventions for Agents

Do not hard-code class names or counts inside model logic; always query from the dataset (e.g., train_dataset.classes).

Use type hints in new Python functions where reasonable.

Wrap runnable scripts with:

if __name__ == "__main__":
    main()


Prefer clear, short functions:

build_model(...)

get_dataloaders(...)

train_one_epoch(...)

evaluate(...)

Logging:

Print epoch number, train/val loss, accuracy, macro-F1 each epoch.

For test evaluation, clearly label as Test and never use test set for model selection.

6. Dataset & Ethics Notes (for any AI agent)

Images come from public web sources and medical references; this project is for education / research only.

The classifier output must never be presented as a medical decision. It is a screening / awareness demo.

Any future UI / API layer should include a clear disclaimer:

“This tool is not a medical device and does not provide diagnosis. Please consult a healthcare professional.”

7. TL;DR for New Agents

Use /final_dataset as read-only source.

Work with /dataset for training & evaluation.

Use PyTorch + pretrained CNN backbone, 4-class classification.

Always compute class-balanced loss and macro-F1.

Add inference.py and grad_cam.py as small, focused scripts.

Keep everything simple enough to later deploy a version on Jetson.