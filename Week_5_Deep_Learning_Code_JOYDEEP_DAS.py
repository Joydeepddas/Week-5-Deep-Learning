# Week 5 - Deep Learning Application in Data Science
# Handwritten Digit Classification using PyTorch
# Prepared by: JOYDEEP DAS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. Reproducibility
# =========================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================================================
# 2. Load the public Digits dataset
# =========================================================
digits = load_digits()

X = digits.data.astype(np.float32)
y = digits.target.astype(np.int64)

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))
print("Classes:", np.unique(y))

# The original pixel values range from 0 to 16.
# Normalize them to 0-1 for neural-network training.
X = X / 16.0

# =========================================================
# 3. Display sample digit images
# =========================================================
fig, axes = plt.subplots(2, 5, figsize=(8, 4))

for digit_class, ax in enumerate(axes.ravel()):
    index = np.where(y == digit_class)[0][0]
    ax.imshow(digits.images[index], cmap="gray_r")
    ax.set_title(f"Digit {digit_class}")
    ax.axis("off")

plt.suptitle("Sample Images from the Digits Dataset")
plt.tight_layout()
plt.show()

# =========================================================
# 4. Class distribution
# =========================================================
class_counts = np.bincount(y, minlength=10)

plt.figure(figsize=(7, 4))
plt.bar(range(10), class_counts)
plt.title("Class Distribution")
plt.xlabel("Digit Class")
plt.ylabel("Number of Samples")
plt.xticks(range(10))
plt.tight_layout()
plt.show()

# =========================================================
# 5. Train / validation / test split
# =========================================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=SEED
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val)

X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# =========================================================
# 6. Create DataLoader
# =========================================================
train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# =========================================================
# 7. Define the neural-network architecture
# =========================================================
class DigitMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Dropout(0.15),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


model = DigitMLP()

print("\nNeural Network Architecture:")
print(model)

# =========================================================
# 8. Loss function and optimizer
# =========================================================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# =========================================================
# 9. Training
# =========================================================
EPOCHS = 100

train_losses = []
val_losses = []
val_accuracies = []

best_val_accuracy = 0.0
best_model_state = None

for epoch in range(EPOCHS):

    # -------------------------
    # Training phase
    # -------------------------
    model.train()

    total_loss = 0.0

    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(batch_X)

    train_loss = total_loss / len(X_train_tensor)

    # -------------------------
    # Validation phase
    # -------------------------
    model.eval()

    with torch.no_grad():

        val_outputs = model(X_val_tensor)

        val_loss = criterion(
            val_outputs,
            y_val_tensor
        ).item()

        val_predictions = val_outputs.argmax(dim=1)

        val_accuracy = (
            val_predictions == y_val_tensor
        ).float().mean().item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Save the best validation model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

# =========================================================
# 10. Restore the best model
# =========================================================
model.load_state_dict(best_model_state)
model.eval()

print(
    f"\nBest Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

# =========================================================
# 11. Plot training and validation loss
# =========================================================
plt.figure(figsize=(7, 4))

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    label="Training Loss"
)

plt.plot(
    range(1, EPOCHS + 1),
    val_losses,
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# 12. Plot validation accuracy
# =========================================================
plt.figure(figsize=(7, 4))

plt.plot(
    range(1, EPOCHS + 1),
    np.array(val_accuracies) * 100
)

plt.title("Validation Accuracy During Training")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.tight_layout()
plt.show()

# =========================================================
# 13. Test-set prediction
# =========================================================
with torch.no_grad():

    test_logits = model(X_test_tensor)

    test_probabilities = torch.softmax(
        test_logits,
        dim=1
    )

    y_pred = test_logits.argmax(
        dim=1
    ).numpy()

# =========================================================
# 14. Evaluation metrics
# =========================================================
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\nFinal Test Results")
print("------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================================================
# 15. Confusion matrix
# =========================================================
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Test-Set Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(range(10))
plt.yticks(range(10))

for i in range(10):
    for j in range(10):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()
plt.show()

# =========================================================
# 16. Per-class F1 scores
# =========================================================
report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

f1_scores = [
    report[str(i)]["f1-score"]
    for i in range(10)
]

plt.figure(figsize=(7, 4))

plt.bar(
    range(10),
    f1_scores
)

plt.title("Per-Class F1 Scores")
plt.xlabel("Digit Class")
plt.ylabel("F1 Score")
plt.xticks(range(10))
plt.ylim(0.85, 1.01)

plt.tight_layout()
plt.show()

# =========================================================
# 17. Display some misclassified examples
# =========================================================
wrong_indices = np.where(
    y_pred != y_test
)[0]

number_to_show = min(
    10,
    len(wrong_indices)
)

if number_to_show > 0:

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(8, 4)
    )

    for ax in axes.ravel():
        ax.axis("off")

    for ax, index in zip(
        axes.ravel(),
        wrong_indices[:number_to_show]
    ):

        ax.imshow(
            X_test[index].reshape(8, 8),
            cmap="gray_r"
        )

        ax.set_title(
            f"True: {y_test[index]} / "
            f"Pred: {y_pred[index]}"
        )

        ax.axis("off")

    plt.suptitle(
        "Examples of Misclassified Test Images"
    )

    plt.tight_layout()
    plt.show()

# =========================================================
# 18. Save dataset copy for GitHub
# =========================================================
dataset_copy = pd.DataFrame(
    X,
    columns=[
        f"pixel_{i:02d}"
        for i in range(64)
    ]
)

dataset_copy["target"] = y

dataset_copy.to_csv(
    "Digits_Week5_DeepLearning.csv",
    index=False
)

print(
    "\nDataset saved as "
    "Digits_Week5_DeepLearning.csv"
)

print("\nWeek 5 deep-learning project completed successfully.")
