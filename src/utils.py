import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(labels, num_classes, device):
    classes = list(range(num_classes))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(classes),
        y=np.array(labels)
    )
    return torch.tensor(weights, dtype=torch.float).to(device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, num_classes=2):
    """
    Evaluate model and return comprehensive metrics.

    Metrics:
        accuracy         - standard accuracy
        weighted_accuracy - balanced accuracy: mean per-class recall.
        f1_macro         - macro F1: unweighted mean of per-class F1.
        mcc              - Matthews Correlation Coefficient [-1, 1].
        roc_auc          - ROC-AUC (binary only). Measures ranking quality
                           independent of the classification threshold.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(logits, dim=1)

            total_loss += loss.item() * x.size(0)
            all_preds  .extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs  .extend(probs.cpu().numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs   = np.array(all_probs)

    accuracy          = (all_preds == all_targets).mean()
    weighted_accuracy = balanced_accuracy_score(all_targets, all_preds)
    f1                = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    mcc               = matthews_corrcoef(all_targets, all_preds)
    roc_auc           = roc_auc_score(all_targets, all_probs[:, 1]) if num_classes == 2 else None

    return {
        "loss":              total_loss / len(loader.dataset),
        "accuracy":          float(accuracy),
        "weighted_accuracy": float(weighted_accuracy),
        "f1_macro":          float(f1),
        "mcc":               float(mcc),
        "roc_auc":           float(roc_auc) if roc_auc is not None else None,
        "preds":             all_preds.tolist(),
        "targets":           all_targets.tolist(),
        "probs":             all_probs.tolist(),
    }


class EarlyStopping:
    """
    Stops training when metric stops improving. Saves best checkpoint.

    Args:
        patience (int): Epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as improvement.
        mode (str): 'max' for F1/accuracy, 'min' for loss.
        checkpoint_path (str): Path to write the best state dict.
    """
    def __init__(self, patience=5, min_delta=1e-4, mode="max",
                 checkpoint_path="best_model.pth"):
        self.patience        = patience
        self.min_delta       = min_delta
        self.mode            = mode
        self.checkpoint_path = checkpoint_path
        self.best_score      = None
        self.counter         = 0

    def __call__(self, score, model):
        improved = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score + self.min_delta)
            or (self.mode == "min" and score < self.best_score - self.min_delta)
        )
        if improved:
            self.best_score = score
            self.counter    = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
        return self.counter >= self.patience

def plot_confusion_matrix(targets, preds, class_names, save_path=None):
    cm   = confusion_matrix(targets, preds, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Normalized Confusion Matrix (row = true class)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_training_curves(history, model_name="", save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=13)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Accuracy: raw vs balanced
    axes[1].plot(epochs, history["val_accuracy"],     label="Val accuracy")
    axes[1].plot(epochs, history["val_weighted_acc"], label="Val balanced acc", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # F1, MCC, AUC
    axes[2].plot(epochs, history["val_f1"],  label="Val F1 (macro)")
    axes[2].plot(epochs, history["val_mcc"], label="Val MCC")
    if any(v > 0 for v in history["val_auc"]):
        axes[2].plot(epochs, history["val_auc"], label="Val AUC", linestyle="--")
    axes[2].set_title("Imbalance-robust metrics")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()