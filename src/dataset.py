import os
import random
from PIL import Image
from torch.utils.data import Dataset

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.JPG'}


class DishwasherDataset(Dataset):
    """
    PyTorch Dataset for the dishwasher-safe image classification task.

    Args:
        file_paths (list[str]): Paths to image files.
        labels (list[int]): Integer class labels corresponding to each file.
        transform: torchvision transform pipeline to apply to each image.
    """

    def __init__(self, file_paths, labels, transform=None):
        assert len(file_paths) == len(labels), \
            f"Mismatch: {len(file_paths)} files but {len(labels)} labels"

        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


def build_file_list(base_dir, class_map, shuffle=True, seed=42):
    """
    Walk the dataset directory and collect image paths with their class labels.
    Skips non-image files (e.g. .DS_Store, Thumbs.db) automatically.

    Args:
        base_dir (str): Root directory containing one subdirectory per class.
        class_map (dict): Maps class folder name → integer label.
                          e.g. {"dishwasher-safe": 1, "not-dishwasher-safe": 0}
        shuffle (bool): Whether to shuffle the file list before returning.
        seed (int): Random seed for reproducible shuffling.

    Returns:
        tuple[list[str], list[int]]: (file_paths, labels)
    """
    file_paths, labels = [], []

    for cls, label in class_map.items():
        class_dir = os.path.join(base_dir, cls)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}. "
                f"Check that 'data_dir' and 'classes' in config.yaml are correct."
            )

        for subdir in os.listdir(class_dir):
            subdir_path = os.path.join(class_dir, subdir)

            if not os.path.isdir(subdir_path):
                continue

            for fname in os.listdir(subdir_path):
                _, ext = os.path.splitext(fname)
                if ext not in SUPPORTED_EXTENSIONS:
                    continue  # skip .DS_Store, Thumbs.db, etc.

                file_paths.append(os.path.join(subdir_path, fname))
                labels.append(label)

    if shuffle:
        combined = list(zip(file_paths, labels))
        random.seed(seed)
        random.shuffle(combined)
        file_paths, labels = zip(*combined)
        file_paths, labels = list(file_paths), list(labels)

    return file_paths, labels