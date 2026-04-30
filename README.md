# Dishwasher-Safe Image Classifier

Binary image classification pipeline that determines whether kitchenware is dishwasher-safe, comparing five CNN architectures on a custom dataset with class imbalance handling and rigorous per-class evaluation.

Part of a three-article series:
- [EDA: Image Dataset Analysis with Pandas and Matplotlib](https://medium.com/@anushree-das/image-dataset-analysis-using-python-libraries-pandas-and-matplotlib-a640e5f59805)
- [Clustering: Finding Structure in Unlabelled Image Data](https://medium.com/@anushree-das/finding-structure-in-unstructured-image-dataset-an-initial-analysis-of-unlabelled-data-0cf2f6231735)

**Dataset:** [Dishwasher-Safe or Not](https://www.kaggle.com/datasets/anushreesitaramdas/dishwasher-safe-or-not) — 1,237 images across 55 sub-classes of kitchenware, collected from Kaggle, Google, and self-photographed items.

---

## Results

All models trained with weighted cross-entropy loss (class weights: dishwasher-safe 1.185, not-dishwasher-safe 0.865), stratified 70/15/15 train/val/test split, and early stopping on macro F1. All pretrained models run with frozen backbone — only the final classifier head was trained. The custom CNN was trained from scratch.

### Test Set Performance

| Model | Backbone | Accuracy | Balanced Acc | Macro F1 | MCC | Epochs |
|---|---|---|---|---|---|---|
| Custom CNN | from scratch | 0.69 | 0.708 | 0.69 | 0.418 | 30 (full) |
| ResNet18 | frozen | 0.69 | 0.702 | 0.69 | 0.400 | 21 (early stop) |
| ResNet50 | frozen | 0.73 | 0.716 | 0.72 | 0.434 | 18 (early stop) |
| **VGG16-BN** | **frozen** | **0.80** | **0.774** | **0.78** | **0.579** | 24 (early stop) |
| AlexNet | frozen | 0.77 | 0.770 | 0.77 | 0.538 | 22 (early stop) |

### Per-Class Breakdown (Test Set)

**Best overall — VGG16-BN:**
```
                     precision  recall  f1-score  support
   dishwasher-safe       0.83    0.64      0.72       78
not-dishwasher-safe      0.78    0.91      0.84      108

           accuracy                        0.80      186
          macro avg       0.81    0.77      0.78      186
```
MCC: 0.579 | Balanced accuracy: 0.774

**Best balanced recall — AlexNet:**
```
                     precision  recall  f1-score  support
   dishwasher-safe       0.72    0.74      0.73       78
not-dishwasher-safe      0.81    0.80      0.80      108

           accuracy                        0.77      186
          macro avg       0.77    0.77      0.77      186
```
MCC: 0.538 | Balanced accuracy: 0.770

### Key observations

**VGG16 is the strongest model overall** by accuracy (0.80), MCC (0.579), and balanced accuracy (0.774), with consistent val and test performance. Its main weakness is lower recall on the dishwasher-safe class (0.64). It's conservative about predicting safe, which is a reasonable failure mode for this domain.

**AlexNet is the most balanced** across both classes (both recalls above 0.74) and trains in a fraction of the time (~1 min/epoch vs ~8 min/epoch for VGG16 on CPU). On a dataset of this size with a frozen backbone, it's a strong practical choice when training time matters.

**ResNet50 showed the largest val-test gap** (val F1: 0.788 → test F1: 0.72) despite having the same frozen backbone setup as VGG16 and AlexNet. This is likely because ResNet50's deeper residual architecture learns more task-specific representations at its final layers, representations that don't transfer as cleanly to this domain as VGG16's fully-connected feature layers. With only 865 training images, there isn't enough data to reliably tune those final representations. Unfreezing the backbone gradually (layer-by-layer) would be the next experiment to try.

**ResNet18 and the custom CNN performed identically on test** (both 0.69 accuracy, MCC ~0.41). The custom CNN training from scratch on 865 images matching a pretrained ResNet18 with a frozen backbone suggests the ImageNet features in ResNet18's shallow architecture aren't meaningfully more informative than what a well-designed custom CNN can learn directly from this domain.

**Custom CNN shows respectable performance given its simplicity** — no pretrained weights, trained from scratch on 865 images, yet competitive with the ResNet variants. This validates the architecture design (three conv blocks, BatchNorm, dropout) and training setup (class weighting, augmentation, ReduceLROnPlateau).

---

## Project structure

```
dishwasher-safe-image-classifier/
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── src/
│   ├── dataset.py               # DishwasherDataset and build_file_list
│   ├── model.py                 # Custom CNN + pretrained model factory
│   ├── train.py                 # Full training pipeline
│   ├── utils.py                 # Metrics, early stopping, plotting
│   └── logger.py                # Dual console + file logger
├── notebooks/
│   ├── 01_eda_dataset_analysis.ipynb   # Class and sub-class distribution EDA
│   └── 02_feature_extraction_clustering.ipynb  # VGG16 + KMeans unsupervised analysis
├── train.py                     # Entry point
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/anushreedas/dishwasher-safe-image-classifier.git
cd dishwasher-safe-image-classifier
pip install -r requirements.txt
```

**Dataset structure expected:**
```
data/final_data/
├── dishwasher-safe/
│   ├── bowl/
│   ├── mug/
│   └── ...
└── not-dishwasher-safe/
    ├── cast_iron_pan/
    ├── knife/
    └── ...
```

---

## Training

**Train with default config (custom CNN):**
```bash
python train.py
```

**Switch model without editing the config:**
```bash
python train.py --model resnet50
python train.py --model vgg16_bn
python train.py --model alexnet
python train.py --model resnet18
```

**Override other hyperparameters at the command line:**
```bash
python train.py --model resnet50 --epochs 20 --lr 0.0005 --batch_size 16
```

**Freeze the backbone (recommended for small datasets):**
```bash
python train.py --model vgg16_bn --freeze_backbone
```
---

## Configuration

All hyperparameters are defined in `configs/config.yaml`. CLI flags override config values without editing the file.

```yaml
model: cnn                  # cnn | resnet18 | resnet50 | vgg16_bn | alexnet
freeze_backbone: false      # freeze all layers except the classifier head
learning_rate: 0.0001
weight_decay: 0.0001
epochs: 30
batch_size: 32
early_stopping_patience: 7
lr_patience: 3              # epochs before ReduceLROnPlateau halves the LR
val_split: 0.15
test_split: 0.15
```
---

## Design decisions

**Class weighting.** The dataset has a 58/42 split (dishwasher-safe vs not). Inverse-frequency weights are computed from the training set only and passed to `CrossEntropyLoss`, so the model penalizes minority-class mistakes proportionally more.

**Stratified splits.** All three splits (train/val/test) are stratified on the class label, preserving the 58/42 ratio in each subset. Without this, rare sub-classes can vanish from the validation or test set entirely.

**Evaluation metrics.** Raw accuracy is reported alongside balanced accuracy, macro F1, and MCC. For a slightly imbalanced dataset, MCC is the single most informative scalar — a model that only predicts the majority class scores near 0, not near 1.

**Frozen backbone for all pretrained models.** With only ~865 training images, fine-tuning all layers of a deep pretrained network risks overfitting. All four pretrained models (ResNet18, ResNet50, VGG16-BN, AlexNet) were run with the backbone frozen — only the final classifier head was trained. This is configurable via `freeze_backbone: true` in config.yaml or the `--freeze_backbone` flag.

**Model-aware image size.** Pretrained ImageNet models use 224×224. The custom CNN uses 256×256 since it has no pretrained weight constraints and benefits from more spatial resolution.

---

## Author

**Anushree Das**
[LinkedIn](https://linkedin.com/in/anushree-s-das) · [GitHub](https://github.com/anushreedas) · [Medium](https://medium.com/@anushree-das)
