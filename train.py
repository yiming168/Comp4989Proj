# train.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score
import numpy as np

DATA_ROOT = "dataset"
BATCH_SIZE = 16
NUM_EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. data preprocessing
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# 2. ImageFolder + DataLoader
train_dataset = datasets.ImageFolder(root=f"{DATA_ROOT}/train",
                                     transform=train_transform)
val_dataset   = datasets.ImageFolder(root=f"{DATA_ROOT}/val",
                                     transform=eval_transform)
test_dataset  = datasets.ImageFolder(root=f"{DATA_ROOT}/test",
                                     transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

class_names = train_dataset.classes
NUM_CLASSES = len(class_names)
print("Detected classes:", class_names)
print("NUM_CLASSES:", NUM_CLASSES)

# 3. calculate class weights for weighted cross-entropy
class_counts = np.bincount(train_dataset.targets)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * len(class_weights)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print("Class counts:", class_counts)
print("Class weights:", class_weights)

# 4. model building and pretraining
def build_model(num_classes: int):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

model = build_model(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 5. train and cv
def run_epoch(loader, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, acc, macro_f1


def main() -> None:
    best_val_f1 = 0.0
    best_model_path = "best_model.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = run_epoch(train_loader, train=True)
        val_loss, val_acc, val_f1 = run_epoch(val_loader, train=False)

        print(f"Epoch {epoch:02d}:")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.3f}, macroF1={train_f1:.3f}")
        print(f"  Val  : loss={val_loss:.4f}, acc={val_acc:.3f}, macroF1={val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  >> New best model saved with val macroF1={best_val_f1:.3f}")

    print("Training done. Best val macroF1:", best_val_f1)

    # 6. evaluate on test set
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_loss, test_acc, test_f1 = run_epoch(test_loader, train=False)
    print(f"Test: loss={test_loss:.4f}, acc={test_acc:.3f}, macroF1={test_f1:.3f}")


if __name__ == "__main__":
    main()
