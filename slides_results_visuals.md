# Slides – Results Visuals (Figures 1–5)

This document turns the guidance in `slidesInstruction.md` into a slide-oriented outline for the 5 key figures, plus minimal code hints to generate each graph.

---

## Slide 1 – Class Distribution (Figure 1)

- **Suggested slide title:** Class Distribution of Nutritional Deficiency Indicators  
- **Figure placeholder:**  
  `![Figure 1 – Class Distribution](figures/figure1_class_distribution.png)`
- **What to show:** Bar chart of image counts per class (pre-augmentation).  
- **Key talking points:**
  - The dataset is imbalanced across the 9 classes (Healthy vs each deficiency).
  - Healthy / control images are relatively more frequent than rare deficiencies.
  - This motivates the use of Weighted Cross-Entropy Loss and Macro-F1.
- **Code sketch (Python):**
  ```python
  import matplotlib.pyplot as plt

  class_names = [...]       # e.g., train_dataset.classes
  class_counts = [...]      # counts per class from your dataset split

  plt.figure(figsize=(10, 6))
  bars = plt.bar(class_names, class_counts, color="skyblue")

  # Optionally highlight Healthy / Control bar
  healthy_index = class_names.index("Unrelated_Control")  # adjust to your label
  bars[healthy_index].set_color("orange")

  plt.xticks(rotation=45, ha="right")
  plt.ylabel("Number of Images")
  plt.title("Class Distribution (Pre-Augmentation)")
  plt.tight_layout()
  plt.savefig("figures/figure1_class_distribution.png", dpi=300)
  ```

---

## Slide 2 – Training vs Validation Metrics (Figure 2)

- **Suggested slide title:** Training and Validation Performance over Epochs  
- **Figure placeholder:**  
  - `![Figure 2A – Loss Curves](figures/figure2a_loss_curves.png)`  
  - `![Figure 2B – Macro-F1 Curves](figures/figure2b_f1_curves.png)`
- **What to show:**  
  - Graph A: Train vs Val Loss per epoch  
  - Graph B: Train vs Val Macro-F1 per epoch
- **Key talking points:**
  - Both training and validation loss decrease and then stabilize (no heavy overfitting).
  - Validation Macro-F1 stabilizes around epoch **[X]**, where Early Stopping kicks in.
  - The close alignment of train/val trends supports good generalization.
- **Code sketch (Python):**
  ```python
  import matplotlib.pyplot as plt

  # Assume you saved these from training
  epochs = list(range(1, len(train_loss) + 1))

  # Loss curves
  plt.figure(figsize=(7, 5))
  plt.plot(epochs, train_loss, label="Train Loss")
  plt.plot(epochs, val_loss, label="Val Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training vs Validation Loss")
  plt.legend()
  plt.tight_layout()
  plt.savefig("figures/figure2a_loss_curves.png", dpi=300)

  # Macro-F1 curves
  plt.figure(figsize=(7, 5))
  plt.plot(epochs, train_f1, label="Train Macro-F1")
  plt.plot(epochs, val_f1, label="Val Macro-F1")
  plt.xlabel("Epoch")
  plt.ylabel("Macro-F1")
  plt.title("Training vs Validation Macro-F1")
  plt.legend()
  plt.tight_layout()
  plt.savefig("figures/figure2b_f1_curves.png", dpi=300)
  ```

---

## Slide 3 – Confusion Matrix (Figure 3)

- **Suggested slide title:** Confusion Matrix on Test Set  
- **Figure placeholder:**  
  `![Figure 3 – Confusion Matrix](figures/figure3_confusion_matrix.png)`
- **What to show:** 9×9 confusion matrix heatmap for the test set.  
- **Key talking points:**
  - Bright diagonal cells indicate correctly classified samples.
  - Off-diagonal cells show confusion between specific deficiency types.
  - Model clearly distinguishes different anatomical regions (eyes vs nails vs tongue).
  - Some confusion remains between clinically similar skin conditions (e.g., between two vitamin-related lesions).
- **Code sketch (Python):**
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.metrics import confusion_matrix

  # y_true, y_pred: arrays of integer class indices from test set
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
  plt.savefig("figures/figure3_confusion_matrix.png", dpi=300)
  ```

---

## Slide 4 – Per-Class F1 Scores (Figure 4)

- **Suggested slide title:** Per-Class F1 Score Comparison  
- **Figure placeholder:**  
  `![Figure 4 – Per-Class F1 Scores](figures/figure4_per_class_f1.png)`
- **What to show:** Horizontal bar chart of F1 score for each class.  
- **Key talking points:**
  - Shows which deficiencies are easiest vs hardest to detect.
  - Weighted Cross-Entropy helps minority classes reach competitive F1 scores.
  - Performance is reasonably balanced instead of dominated by Healthy / majority class.
- **Code sketch (Python):**
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.metrics import f1_score

  # Per-class F1 scores on test set
  f1_scores = f1_score(y_true, y_pred, average=None)

  plt.figure(figsize=(10, 6))
  sns.barplot(x=f1_scores, y=class_names, palette="viridis")
  plt.title("Per-Class F1 Score (Test Set)")
  plt.xlabel("F1 Score")
  plt.xlim(0.0, 1.0)
  plt.tight_layout()
  plt.savefig("figures/figure4_per_class_f1.png", dpi=300)
  ```

---

## Slide 5 – Grad-CAM Interpretability (Figure 5)

- **Suggested slide title:** Grad-CAM Visualization of Model Attention  
- **Figure placeholder:**  
  `![Figure 5 – Grad-CAM](figures/figure5_gradcam_grid.png)`
- **What to show:** Grid of image pairs (original vs Grad-CAM heatmap overlay) for several classes (e.g., Koilonychia, Bitot’s spots, Glossitis).  
- **Key talking points:**
  - The model focuses on clinically relevant regions (nail plate depression, scleral lesions, tongue surface) rather than background.
  - Confirms that the network’s decision-making aligns with medical intuition.
  - Supports the Explainable AI (XAI) component of the project.
- **Code sketch (Python, high-level):**
  ```python
  # Pseudocode – assumes you already have a Grad-CAM helper implemented
  import matplotlib.pyplot as plt

  images = [...]         # list of input tensors or file paths
  class_indices = [...]  # target class index for each image

  fig, axes = plt.subplots(len(images), 2, figsize=(8, 4 * len(images)))

  for i, (img, cls_idx) in enumerate(zip(images, class_indices)):
      original = load_numpy_or_pil(img)              # your own loader
      gradcam = generate_gradcam_heatmap(img, cls_idx)  # your Grad-CAM function

      axes[i, 0].imshow(original)
      axes[i, 0].set_title("Original")
      axes[i, 0].axis("off")

      axes[i, 1].imshow(overlay_heatmap(original, gradcam))
      axes[i, 1].set_title("Grad-CAM")
      axes[i, 1].axis("off")

  plt.tight_layout()
  plt.savefig("figures/figure5_gradcam_grid.png", dpi=300)
  ```

---

## Suggested Slide Order

1. Brief model summary (no figure, optional).  
2. **Figure 1 – Class Distribution** (motivation: imbalance).  
3. **Figure 2 – Training vs Validation Metrics** (training behavior).  
4. **Figure 3 – Confusion Matrix** (overall performance on test set).  
5. **Figure 4 – Per-Class F1 Scores** (balanced performance).  
6. **Figure 5 – Grad-CAM** (explainability and medical relevance).

