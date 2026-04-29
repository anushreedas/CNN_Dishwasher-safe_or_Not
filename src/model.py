import torch.nn as nn
from torchvision import models


class CNN(nn.Module):
    """
    Custom two-block CNN baseline for binary image classification.

    Architecture:
        Block 1: Conv(3→64, 5x5) → BN → ReLU → MaxPool(2)
        Block 2: Conv(64→128, 3x3) → BN → ReLU → MaxPool(2)
        Block 3: Conv(128→256, 3x3) → BN → ReLU → MaxPool(2)
        Head:    AdaptiveAvgPool → Flatten → Dropout(0.5) → Linear

    Args:
        num_classes (int): Number of output classes.
        dropout (float): Dropout rate before the final linear layer.
    """

    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),  # preserve more spatial info than (1,1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(model_name, num_classes, freeze_backbone=False):
    """
    Build a pretrained torchvision model with its classification head replaced.

    Supported model names: 'resnet18', 'resnet50', 'vgg16_bn', 'alexnet', 'cnn'

    Args:
        model_name (str): Name of the model architecture.
        num_classes (int): Number of output classes.
        freeze_backbone (bool): If True, freeze all layers except the final
                                classifier head.

    Returns:
        nn.Module: Model ready for training.
    """
    model_name = model_name.lower()

    if model_name == "cnn":
        return CNN(num_classes=num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "vgg16_bn":
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    elif model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: cnn, resnet18, resnet50, vgg16_bn, alexnet"
        )

    if freeze_backbone:
        # Freeze everything except the final classifier head
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier.6" not in name:
                param.requires_grad = False

    return model