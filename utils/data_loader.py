import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import os
import pandas as pd
from PIL import Image

class KaggleCIFAR10Dataset(Dataset):
    """Custom dataset for CIFAR-10 in train/test + CSV format."""

    def __init__(self, data_dir, is_train=True, transform=None):
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform

        # Class name mapping (same for train and test)
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
        self.label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Common root for labels and image folders
        labels_root = os.path.join(data_dir, 'raw')

        if is_train:
            labels_path = os.path.join(labels_root, 'trainLabels.csv')
            self.image_dir = os.path.join(labels_root, 'train')
        else:
            labels_path = os.path.join(labels_root, 'testLabels.csv')
            self.image_dir = os.path.join(labels_root, 'test')

        self.labels_df = pd.read_csv(labels_path)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        image_id = row['id']
        label_name = row['label']

        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(image_path).convert('RGB')
        label = self.label_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_cifar10_loaders(data_dir='./data', batch_size=64, num_workers=4, val_split=0.1):
    """
    Load CIFAR-10 from custom directory layout with augmentation and splits.

    Args:
        data_dir: Directory containing 'raw/train', 'raw/test', and both CSV files
        batch_size: Batch size for loaders
        num_workers: Worker processes for DataLoader
        val_split: Fraction of training data used for validation

    Returns:
        (train_loader, val_loader, test_loader)
    """
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Full training dataset
    full_train_dataset = KaggleCIFAR10Dataset(
        data_dir=data_dir,
        is_train=True,
        transform=train_transform
    )

    # Split into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Use eval transforms for validation set
    val_dataset = KaggleCIFAR10Dataset(
        data_dir=data_dir,
        is_train=True,
        transform=eval_transform
    )
    val_subset.dataset = val_dataset

    # Load labeled test dataset
    test_dataset = KaggleCIFAR10Dataset(
        data_dir=data_dir,
        is_train=False,
        transform=eval_transform
    )

    # DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"📊 Dataset loaded successfully!")
    print(f"   Training samples: {len(train_subset)}")
    print(f"   Validation samples: {len(val_subset)}")
    print(f"   Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def get_cifar10_info():
    """Return CIFAR-10 dataset information."""
    return {
        'num_classes': 10,
        'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck'],
        'input_shape': (3, 32, 32),
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    }
