import os
import cv2
from typing import List

import numpy as np
import pandas as pd
import torch.utils.data
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from PIL import Image


def get_all_filenames(
        path: str
) -> List[str]:
    all_file_names = []
    for _, _, field_names in os.walk(path):
        all_file_names.extend([f for f in field_names if 'jpg' in f])
    return all_file_names


class ProjectDataSet(Dataset):
    def __init__(
            self,
            image_folder_path: str,
            data_label_path: str = None,
            is_training=True,
            is_superclass=True,
            img_size=8
    ):
        self._image_folder_path = image_folder_path
        self._data_label_path = data_label_path
        self._file_names = get_all_filenames(image_folder_path)
        self._is_training = is_training
        self._is_superclass = is_superclass
        self._is_training = data_label_path is not None
        self._img_size = img_size

        self._label_dict = self._get_class_label(
            data_label_path
        )

    def _get_class_label(
            self,
            data_label_path: str
    ):
        if data_label_path:
            y_train_label = pd.read_csv(data_label_path)
            class_column_name = 'superclass_index' if self._is_superclass else 'subclass_index'
            if class_column_name not in y_train_label.columns:
                raise RuntimeError(f'Required column {class_column_name} is missing')
            if 'image' not in y_train_label.columns:
                raise RuntimeError('Required column image_index is missing')
            return dict(
                (t.image, getattr(t, class_column_name))
                for t in y_train_label.itertuples()
            )
        return dict()

    def _get_transformers(self):
        if self._is_training:
            return transforms.Compose([
                transforms.Resize(self._img_size),
                transforms.RandomCrop(self._img_size, padding=self._img_size // 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4707, 0.4431, 0.3708), (0.1577, 0.1587, 0.1783)),
            ])

        return transforms.Compose([
            transforms.Resize(self._img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4707, 0.4431, 0.3708), (0.1577, 0.1587, 0.1783)),
        ])

    def __getitem__(self, idx):
        image_filepath = os.path.join(self._image_folder_path, self._file_names[idx])
        img = cv2.imread(image_filepath)  # Read in image from filepath.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self._get_transformers()(img)
        return img, self._label_dict.get(self._file_names[idx], 0)

    def __len__(self):
        return len(self._file_names)


class CifarValidationDataset(Dataset):
    def __init__(self, img_size=8, cifar_data_folder='./data', download=True):
        self._img_size = img_size
        self._cifar_data_folder = cifar_data_folder
        self._download = download
        self._cifar_dataset = torch.utils.data.ConcatDataset([
            self._get_cifar10_dataset(),
            self._get_cifar100_dataset()
        ])

    def __getitem__(self, idx):
        return self._cifar_dataset[idx]

    def __len__(self):
        return len(self._cifar_dataset)

    def _get_cifar10_dataset(self) -> CIFAR10:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                *((0.49139968, 0.48215827, 0.44653124),
                  (0.24703233, 0.24348505, 0.26158768)))
        ])
        # Download the CIFAR10 Data
        cifar10_train = torchvision.datasets.CIFAR10(
            root=self._cifar_data_folder,
            train=True,
            download=self._download,
            transform=transform
        )
        cifar10_test = torchvision.datasets.CIFAR10(
            root=self._cifar_data_folder,
            train=False,
            download=self._download,
            transform=transform
        )

        x_train = cifar10_train.data
        y_train = np.asarray(cifar10_train.targets)

        x_test = cifar10_test.data
        y_test = np.asarray(cifar10_test.targets)

        bird_data = np.concatenate(
            [x_train[np.squeeze(y_train == 2)], x_test[np.squeeze(y_test == 2)]])
        dog_data = np.concatenate(
            [x_train[np.squeeze(y_train == 5)], x_test[np.squeeze(y_test == 5)]])

        dog_data = dog_data[np.random.choice(len(dog_data), size=3000, replace=False)]
        bird_data = bird_data[np.random.choice(len(bird_data), size=3000, replace=False)]

        y = np.repeat([[0], [1]], repeats=[3000], axis=-1).flatten()

        cifar10_train.data = np.concatenate([bird_data, dog_data])
        cifar10_train.targets = y

        return cifar10_train

    def _get_cifar100_dataset(self) -> CIFAR100:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)))
        ])
        # Download the CIFAR100 Data
        cifar100_train = torchvision.datasets.CIFAR100(
            root=self._cifar_data_folder,
            train=True,
            download=self._download,
            transform=transform
        )
        cifar100_test = torchvision.datasets.CIFAR100(
            root=self._cifar_data_folder,
            train=False,
            download=self._download,
            transform=transform
        )

        x_train = cifar100_train.data
        y_train = np.asarray(cifar100_train.targets)

        x_test = cifar100_test.data
        y_test = np.asarray(cifar100_test.targets)

        reptile_data = np.concatenate(
            [x_train[np.squeeze(y_train == 15)], x_test[np.squeeze(y_test == 15)]])

        y = np.repeat([[2]], repeats=[3000], axis=-1).flatten()

        cifar100_train.data = reptile_data
        cifar100_train.targets = y

        return cifar100_train
