import os
import pickle
from pathlib import Path
from typing import Union

import pandas as pd
import sklearn
import sklearn.preprocessing
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base class for datasets."""

    IMG_COL = "image"
    LBL_COL = "label"

    def __init__(self, transform=None, val_transform=None, **kwargs):
        """
        Initialize the dataset.

        Sets the correct path for the needed arguments.

        Parameters
        ----------
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        """
        super().__init__()
        self.training = True
        self.transform = transform
        self.val_transform = val_transform
        self.meta_data = pd.DataFrame()
        self.labelencoder = sklearn.preprocessing.LabelEncoder()

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def save_label_encoder(self, path: str):
        if self.labelencoder is not None:
            le_file_name = os.path.join(path, "label_encoder.pickle")
            le_file = open(le_file_name, "wb")
            pickle.dump(self.labelencoder, le_file)
            le_file.close()

    def check_path(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path needs to exist: {path}")
        return path

    @staticmethod
    def find_files_with_extension(directory_path: Union[str, Path], extension: str):
        extension = extension.replace("*", "")
        l_files = []
        for entry in os.scandir(directory_path):
            if entry.is_file() and entry.name.endswith(extension):
                l_files.append(entry.path)
            elif entry.is_dir():
                l_files.extend(
                    BaseDataset.find_files_with_extension(
                        directory_path=entry.path,
                        extension=extension,
                    )
                )
        return l_files

    @staticmethod
    def collate_fn(batch):
        return torch.utils.data.dataloader.default_collate(batch)
