import os
import cv2
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
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
            transform=None,
            is_superclass=True
    ):
        self._image_folder_path = image_folder_path
        self._data_label_path = data_label_path
        self._file_names = get_all_filenames(image_folder_path)
        self._transform = transform
        self._is_superclass = is_superclass
        self._is_training = data_label_path is not None

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

    def __getitem__(self, idx):
        image_filepath = os.path.join(self._image_folder_path, self._file_names[idx])
        img = cv2.imread(image_filepath)  # Read in image from filepath.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self._transform:
            img = self._transform(img)
        return img, self._label_dict.get(self._file_names[idx], 0)

    def __len__(self):
        return len(self._file_names)


class ExtractedCifarDataset(Dataset):
    stats = ((0.49303856, 0.47863943, 0.4250731), (0.24740638, 0.23635836, 0.24916491))

    def __init__(self, data_folder_path, train=True, transform=None):

        self.data_folder_path = data_folder_path
        if train:
            data_path = os.path.join(data_folder_path, 'train_data.npy')
            target_path = os.path.join(data_folder_path, 'train_targets.npy')
        else:
            data_path = os.path.join(data_folder_path, 'test_data.npy')
            target_path = os.path.join(data_folder_path, 'test_targets.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.classes = ['bird', 'dog', 'repitle']
        self.class_to_idx = {'bird': 0, 'dog': 1, 'reptile': 2}
        self.transform = transform

    def __getitem__(self, idx):
        sample = Image.fromarray(np.uint8(self.data[idx])).convert('RGB')
        if self.transform:
            sample = self.transform(sample)
        return sample, self.targets[idx]

    def __len__(self):
        return len(self.data)
