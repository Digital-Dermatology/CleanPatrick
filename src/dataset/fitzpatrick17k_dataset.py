import os
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .base_dataset import BaseDataset


class FitzpatrickLabel(Enum):
    # col name of label code, col name of label
    HIGH = "lbl_high", "three_partition_label"
    MID = "lbl_mid", "nine_partition_label"
    LOW = "lbl_low", "granular_partition_label"


class Fitzpatrick17kDataset(BaseDataset):
    """Fitzpatrick17k dataset."""

    IMG_COL = "path"
    LBL_COL = "lbl_high"

    def __init__(
        self,
        csv_file: Union[str, Path] = "data/fitzpatrick17k/fitzpatrick17k.csv",
        dataset_dir: Union[str, Path] = "data/fitzpatrick17k/",
        transform=None,
        val_transform=None,
        label_col: FitzpatrickLabel = FitzpatrickLabel.HIGH,
        high_quality: bool = False,
        return_fitzpatrick: bool = False,
        return_path: bool = False,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Sets the correct path for the needed arguments.

        Parameters
        ----------
        csv_file : str
            Path to the csv file with metadata, including annotations.
        dataset_dir : str
            Directory with all the images.
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        val_transform : Union[callable, optional]
            Optional transform to be applied to the images when in validation mode.
        label_col : FitzpatrickLabel
            Which of the different partition labels should be used.
        high_quality : bool
            If only high quality samples should be used, e.g. the ones annotated by expert dermatologists.
        return_fitzpatrick : bool
            If the fitzpatrick type of the image should be returned or not.
        return_path : bool
            If the path of the image should be returned or not.
        """
        super().__init__(transform=transform, val_transform=val_transform, **kwargs)
        # check if the dataset path exists
        self.csv_file = Path(csv_file)
        if not self.csv_file.exists():
            raise ValueError(f"CSV metadata path must exist, path: {self.csv_file}")
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Image path must exist, path: {self.dataset_dir}")
        # transform the dataframe for better loading
        self.imageid_path_dict = {
            os.path.splitext(os.path.basename(x))[0]: x
            for x in glob(os.path.join(self.dataset_dir, "*", "*.jpg"))
        }
        # load the metadata
        self.meta_data = pd.DataFrame(pd.read_csv(csv_file, index_col=0))
        self.meta_data["path"] = self.meta_data["md5hash"].map(
            self.imageid_path_dict.get
        )
        # transform the string labels into categorical values
        self.meta_data.rename(
            columns={"label": FitzpatrickLabel.LOW.value[1]},
            inplace=True,
        )
        self.meta_data[FitzpatrickLabel.LOW.value[1]] = self.meta_data[
            FitzpatrickLabel.LOW.value[1]
        ].replace(np.nan, "undefined")
        self.meta_data[FitzpatrickLabel.MID.value[1]] = self.meta_data[
            FitzpatrickLabel.MID.value[1]
        ].replace(np.nan, "undefined")
        self.meta_data[FitzpatrickLabel.HIGH.value[1]] = self.meta_data[
            FitzpatrickLabel.HIGH.value[1]
        ].replace(np.nan, "undefined")

        self.meta_data[FitzpatrickLabel.LOW.value[0]] = pd.factorize(
            self.meta_data[FitzpatrickLabel.LOW.value[1]]
        )[0]
        self.meta_data[FitzpatrickLabel.MID.value[0]] = pd.factorize(
            self.meta_data[FitzpatrickLabel.MID.value[1]]
        )[0]
        self.meta_data[FitzpatrickLabel.HIGH.value[0]] = pd.factorize(
            self.meta_data[FitzpatrickLabel.HIGH.value[1]]
        )[0]

        # select only high quality samples if requested
        if high_quality:
            self.meta_data = self.meta_data[self.meta_data["qc"].notna()]
        self.meta_data.reset_index(drop=True, inplace=True)

        # global configs
        self.return_fitzpatrick = return_fitzpatrick
        self.return_path = return_path
        self.LBL_COL = label_col.value[0]
        self.classes = self.meta_data[label_col.value[1]].unique().tolist()
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.meta_data.loc[self.meta_data.index[idx], self.IMG_COL]
        image = Image.open(img_path)
        image = image.convert("RGB")
        if self.transform and self.training:
            image = self.transform(image)
        elif self.val_transform and not self.training:
            image = self.val_transform(image)

        diagnosis = self.meta_data.loc[self.meta_data.index[idx], self.LBL_COL]
        if self.return_path:
            return image, img_path, int(diagnosis)
        elif self.return_fitzpatrick:
            return image, int(
                self.meta_data.loc[self.meta_data.index[idx], "fitzpatrick"]
            )
        else:
            return image, int(diagnosis)
