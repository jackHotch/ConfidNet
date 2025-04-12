import torch
from torchvision import datasets
from torchvision.datasets import ImageFolder
import os

from confidnet.augmentations import get_composed_augmentations
from confidnet.loaders.camvid_dataset import CamvidDataset
from torchvision.datasets import EuroSAT
from torchvision.datasets import STL10
from torch.utils.data import random_split
from confidnet.loaders.loader import AbstractDataLoader

class EuroSATLoader(AbstractDataLoader):
    def load_dataset(self):
        self.augmentations_train = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize to 64x64
            transforms.ToTensor(),
            # Add any other augmentation steps you want here, like random crop or horizontal flip
        ])


        self.augmentations_test = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize to 64x64
            transforms.ToTensor(),
        ])
       
        self.train_dataset = EuroSAT(
            root=self.data_dir, download=True, transform=self.augmentations_train
        )
        self.test_dataset = EuroSAT(
            root=self.data_dir, download=True, transform=self.augmentations_test
        )

class STL10Loader(AbstractDataLoader):
    def load_dataset(self):
        # Load full train split with augmentations
        full_train_dataset = STL10(
            root=self.data_dir,
            split="train",
            download=True,
            transform=self.augmentations_train
        )

        total_size = len(full_train_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_train_dataset, [train_size, val_size, test_size]
        )

        # Load official test set (can optionally replace `self.test_dataset`)
        self.official_test_dataset = STL10(
            root=self.data_dir,
            split="test",
            download=True,
            transform=self.augmentations_test
        )

        # Override transforms for val/test
        self.val_dataset.dataset.transform = self.augmentations
        self.test_dataset.dataset.transform = self.augmentations_test

    def make_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

class FashionMNISTLoader(AbstractDataLoader):
    def load_dataset(self):
        self.train_dataset = datasets.FashionMNIST(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        self.test_dataset = datasets.FashionMNIST(
            root=self.data_dir, train=False, download=True, transform=self.augmentations_test
        )

import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import os

from confidnet.augmentations import get_composed_augmentations
from confidnet.loaders.loader import AbstractDataLoader


class ImagenetteLoader(AbstractDataLoader):
    def load_dataset(self):
        # Expected directory structure:
        # data_dir/
        # └── imagenette/
        #     ├── train/
        #     │   ├── class1/
        #     │   └── ...
        #     └── val/
        #         ├── class1/
        #         └── ...
        train_dir = os.path.join(self.data_dir, "data/train")
        val_dir = os.path.join(self.data_dir, "data/val")

        # Load datasets using torchvision's ImageFolder
        self.train_dataset = ImageFolder(
            root=train_dir,
            transform=self.augmentations_train
        )
        self.test_dataset = ImageFolder(
            root=val_dir,
            transform=self.augmentations_test
        )


class MNISTLoader(AbstractDataLoader):
    def load_dataset(self):
        self.train_dataset = datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        self.test_dataset = datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=self.augmentations_test
        )


class SVHNLoader(AbstractDataLoader):
    def load_dataset(self):
        self.train_dataset = datasets.SVHN(
            root=self.data_dir, split="train", download=True, transform=self.augmentations_train
        )
        self.test_dataset = datasets.SVHN(
            root=self.data_dir, split="test", download=True, transform=self.augmentations_test
        )


class CIFAR10Loader(AbstractDataLoader):
    def load_dataset(self):
        self.train_dataset = datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.augmentations_test
        )


class CIFAR100Loader(AbstractDataLoader):
    def load_dataset(self):
        self.train_dataset = datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        self.test_dataset = datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=self.augmentations_test
        )


class CamVidLoader(AbstractDataLoader):
    def add_augmentations(self):
        self.augmentations_train = get_composed_augmentations(
            self.augmentations, training="segmentation"
        )
        self.augmentations_val = get_composed_augmentations(
            {
                key: self.augmentations[key]
                for key in self.augmentations
                if key in ["normalize", "resize"]
            },
            verbose=False,
            training="segmentation",
        )
        self.augmentations_test = get_composed_augmentations(
            {key: self.augmentations[key] for key in self.augmentations if key == "normalize"},
            verbose=False,
            training="segmentation",
        )

    def load_dataset(self):
        # Loading dataset
        self.train_dataset = CamvidDataset(
            data_dir=self.data_dir, split="train", transform=self.augmentations_train
        )
        self.val_dataset = CamvidDataset(
            data_dir=self.data_dir, split="val", transform=self.augmentations_val
        )
        self.test_dataset = CamvidDataset(
            data_dir=self.data_dir, split="test", transform=self.augmentations_test
        )

    def make_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
