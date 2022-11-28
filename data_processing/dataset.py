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
            if 'image_index' not in y_train_label.columns:
                raise RuntimeError('Required column image_index is missing')
            return dict(
                (t.image_index, getattr(t, class_column_name))
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
