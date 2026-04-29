import os
import json
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    balanced_accuracy_score
)

from src.dataset import DishwasherDataset, build_file_list
from src.logger import get_logger
from src.model import build_model
from src.utils import (
    get_class_weights,
    train_one_epoch,
    evaluate,
    EarlyStopping,
    plot_confusion_matrix,
    plot_training_curves,
)

logger = get_logger(__name__)

# Pretrained models were trained on 224x224 ImageNet images.
# The custom CNN has no such constraint — 256x256.
MODEL_IMG_SIZES = {
    "resnet18":  (224, 224),
    "resnet50":  (224, 224),
    "vgg16_bn":  (224, 224),
    "alexnet":   (224, 224),
    "cnn":       (256, 256),
}


def get_img_size(model_name, config):
    """
    Return the correct image size for a given model.

    Priority:
        1. Explicit override in config ("img_size" key)
        2. Model-specific default from MODEL_IMG_SIZES
        3. Fall back to (224, 224) for unknown models

    Args:
        model_name (str): Model architecture name.
        config (dict): Config dict loaded from YAML.

    Returns:
        tuple[int, int]: (height, width)
    """
    if "img_size" in config and config["img_size"] is not None:
        size = tuple(config["img_size"])
        logger.info("[img_size] Using config override: %s", size)
        return size

    size = MODEL_IMG_SIZES.get(model_name.lower(), (224, 224))
    logger.info("[img_size] Using model default for '%s': %s", model_name, size)
    return size


def get_transforms(img_size, augment=False):
    """
    Build the image transform pipeline.

    Training transforms apply random augmentation to improve generalization
    and increase effective dataset size for underrepresented sub-classes.

    Args:
        img_size (tuple[int, int]): Target (height, width).
        augment (bool): If True, apply random augmentation (training only).

    Returns:
        torchvision.transforms.Compose
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if augment:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize,
        ])

def print_classification_report(targets, preds, class_names):
    logger.info(classification_report(targets, preds, target_names=class_names, zero_division=0))
    logger.info(f"MCC:               {matthews_corrcoef(targets, preds):.4f}")
    logger.info(f"Balanced accuracy: {balanced_accuracy_score(targets, preds):.4f}")


def run_training(config):
    """
    Full training pipeline

    Split strategy:
        Train (70%) - gradient updates only
        Val   (15%) - early stopping and LR scheduling
        Test  (15%) - held out completely, evaluated exactly once at the end

    Args:
        config (dict): Loaded from configs/config.yaml.

    Returns:
        tuple: (trained model, training history dict)
    """
    logger.info("Effective config:\n%s", json.dumps(config, indent=2))
    # Device: prefer CUDA > Apple MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    model_name  = config.get("model", "cnn")
    num_classes = len(config["classes"])
    class_names = list(config["classes"].keys())
    seed        = config.get("seed", 42)

    # -----------------------------------------------------------------------
    # 1. Load file paths and labels
    # -----------------------------------------------------------------------
    file_paths, labels = build_file_list(
        config["data_dir"],
        config["classes"],
        shuffle=True,
        seed=seed
    )
    logger.info("Total images: %d", len(file_paths))

    # -----------------------------------------------------------------------
    # 2. Three-way stratified split: train / val / test
    #
    #    Stratify at every step so class ratios are preserved in all three
    #    subsets. Without this, rare sub-classes can vanish from val/test.
    #
    #    Step a: hold out test (15% of total)
    #    Step b: split remainder into train (70%) and val (15%)
    # -----------------------------------------------------------------------
    val_size  = config.get("val_split",  0.15)
    test_size = config.get("test_split", 0.15)

    # Step a
    train_val_paths, test_paths, y_train_val, y_test = train_test_split(
        file_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )

    # Step b — val fraction is relative to the remaining (train + val) pool
    val_fraction_of_remainder = val_size / (1.0 - test_size)

    train_paths, val_paths, y_train, y_val = train_test_split(
        train_val_paths, y_train_val,
        test_size=val_fraction_of_remainder,
        stratify=y_train_val,
        random_state=seed
    )

    total = len(file_paths)
    logger.info(
        "Split | Train: %d (%.0f%%) | Val: %d (%.0f%%) | Test: %d (%.0f%%)",
        len(train_paths), len(train_paths)/total*100,
        len(val_paths),   len(val_paths)/total*100,
        len(test_paths),  len(test_paths)/total*100,
    )

    # -----------------------------------------------------------------------
    # 3. Image size and transforms
    # -----------------------------------------------------------------------
    img_size = get_img_size(model_name, config)

    train_dataset = DishwasherDataset(train_paths, y_train, transform=get_transforms(img_size, augment=True))
    val_dataset   = DishwasherDataset(val_paths,   y_val,   transform=get_transforms(img_size, augment=False))
    test_dataset  = DishwasherDataset(test_paths,  y_test,  transform=get_transforms(img_size, augment=False))

    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 2),
        pin_memory=(device.type == "cuda")
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    # -----------------------------------------------------------------------
    # 4. Model
    # -----------------------------------------------------------------------
    model = build_model(
        model_name,
        num_classes=num_classes,
        freeze_backbone=config.get("freeze_backbone", False)
    ).to(device)
    logger.info("Model: %s | Classes: %d | Img size: %s", model_name, num_classes, img_size)

    # -----------------------------------------------------------------------
    # 5. Loss with class weighting
    #
    #    Computed from TRAINING labels only — never val or test — to avoid
    #    data leakage into the loss function.
    # -----------------------------------------------------------------------
    class_weights = get_class_weights(y_train, num_classes, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    logger.info("Class weights: %s", {n: f"{w:.3f}" for n, w in zip(class_names, class_weights.cpu().numpy())})

    # -----------------------------------------------------------------------
    # 6. Optimizer and LR scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4)
    )
    # ReduceLROnPlateau monitors val F1 and halves LR when it plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=config.get("lr_patience", 3),
        verbose=True
    )

    # -----------------------------------------------------------------------
    # 7. Early stopping
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(config["model_path"]), exist_ok=True)
    checkpoint_path = config["model_path"].replace(".pth", f"_{model_name}_best.pth")

    early_stopping = EarlyStopping(
        patience=config.get("early_stopping_patience", 7),
        mode="max",
        checkpoint_path=checkpoint_path
    )

    # -----------------------------------------------------------------------
    # 8. Training loop
    # -----------------------------------------------------------------------
    history = {
        "train_loss":    [],
        "val_loss":      [],
        "val_accuracy":  [],
        "val_weighted_acc": [],
        "val_f1":        [],
        "val_mcc":       [],
        "val_auc":       [],
    }

    for epoch in range(config["epochs"]):
        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, num_classes=num_classes)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_weighted_acc"].append(val_metrics["weighted_accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])
        history["val_mcc"].append(val_metrics["mcc"])
        history["val_auc"].append(val_metrics.get("roc_auc", 0.0))

        logger.info(
            "Epoch %3d/%d | Train Loss: %.4f | Val Loss: %.4f | "
            "Acc: %.4f | W-Acc: %.4f | F1: %.4f | MCC: %.4f | AUC: %.4f",
            epoch + 1, config["epochs"],
            train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["weighted_accuracy"],
            val_metrics["f1_macro"],
            val_metrics["mcc"],
            val_metrics.get("roc_auc") or 0.0,
        )

        scheduler.step(val_metrics["f1_macro"])

        if early_stopping(val_metrics["f1_macro"], model):
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    # -----------------------------------------------------------------------
    # 9. Final evaluation
    # -----------------------------------------------------------------------
    logger.info("Loaded best checkpoint (val F1: %.4f)", early_stopping.best_score)

    logger.info("--- Validation Set (final) ---")
    val_final = evaluate(model, val_loader, device, num_classes=num_classes)
    print_classification_report(val_final["targets"], val_final["preds"], class_names)

    # Test is evaluated exactly once
    logger.info("--- Test Set (held-out, evaluated once) ---")
    test_metrics = evaluate(model, test_loader, device, num_classes=num_classes)
    print_classification_report(test_metrics["targets"], test_metrics["preds"], class_names)

    plot_confusion_matrix(
        test_metrics["targets"], test_metrics["preds"], class_names,
        save_path=f"models/{model_name}_test_confusion_matrix.png"
    )
    plot_training_curves(
        history, model_name=model_name,
        save_path=f"models/{model_name}_curves.png"
    )

    # -----------------------------------------------------------------------
    # 10. Save model
    # -----------------------------------------------------------------------
    torch.save({
        "model_name":       model_name,
        "model_state_dict": model.state_dict(),
        "class_map":        config["classes"],
        "img_size":         list(img_size),
        "val_f1":           val_final["f1_macro"],
        "val_mcc":          val_final["mcc"],
        "val_auc":          val_final.get("roc_auc"),
        "test_f1":          test_metrics["f1_macro"],
        "test_accuracy":    test_metrics["accuracy"],
        "test_weighted_acc": test_metrics["weighted_accuracy"],
        "test_mcc":         test_metrics["mcc"],
        "test_auc":         test_metrics.get("roc_auc"),
    }, config["model_path"])

    logger.info("Model saved to %s", config["model_path"])
    return model, history