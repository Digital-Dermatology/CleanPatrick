import itertools
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.dataset.fitzpatrick17k_dataset import Fitzpatrick17kDataset, FitzpatrickLabel


class CleanPatrick(Fitzpatrick17kDataset):
    """Base class for datasets."""

    def __init__(
        self,
        cleanpatrick_dataset_dir: Union[str, Path] = "data/CleanPatrick/",
        fitzpatrick_csv_file: Union[
            str, Path
        ] = "data/fitzpatrick17k/fitzpatrick17k.csv",
        fitzpatrick_dataset_dir: Union[str, Path] = "data/fitzpatrick17k/",
        transform=None,
        val_transform=None,
        return_meta: bool = False,
        return_path: bool = False,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Sets the correct path for the needed arguments.

        Parameters
        ----------
        cleanpatrick_dataset_dir : str
            Directory with all the files of CleanPatrick.
        fitzpatrick_csv_file : str
            Path to the csv file of Fitzpatrick17k with metadata, including annotations.
        fitzpatrick_dataset_dir : str
            Directory with all the images of Fitzpatrick17k.
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        val_transform : Union[callable, optional]
            Optional transform to be applied to the images when in validation mode.
        return_fitzpatrick : bool
            If the fitzpatrick type of the image should be returned or not.
        return_path : bool
            If the path of the image should be returned or not.
        """
        super().__init__(
            csv_file=fitzpatrick_csv_file,
            dataset_dir=fitzpatrick_dataset_dir,
            transform=transform,
            val_transform=val_transform,
            label_col=FitzpatrickLabel.LOW,
            return_path=return_path,
        )
        self.return_meta = return_meta
        self.cleanpatrick_dataset_dir = Path(cleanpatrick_dataset_dir)
        self.off_topic_path = self.cleanpatrick_dataset_dir / "off_topic_samples.csv"
        self.near_duplicates = self.cleanpatrick_dataset_dir / "near_duplicates.csv"
        self.label_errors = self.cleanpatrick_dataset_dir / "label_errors.csv"
        for path in [self.off_topic_path, self.near_duplicates, self.label_errors]:
            if not path.exists():
                raise ValueError(f"CSV file must exist, path: {path}")

        self.df_ot = pd.read_csv(self.off_topic_path)
        self.df_nd = pd.read_csv(self.near_duplicates)
        self.df_lb = pd.read_csv(self.label_errors)

        # merge the meta data with the hashes
        self.df_ot["path"] = self.df_ot["md5hash"].map(self.imageid_path_dict.get)
        self.df_lb["path"] = self.df_lb["md5hash"].map(self.imageid_path_dict.get)
        self.df_nd["md5hash_list"] = self.df_nd["md5hash_list"].apply(
            lambda x: [x.strip() for x in x.split(",")]
        )
        self.df_nd["path_list"] = self.df_nd["md5hash_list"].apply(
            lambda x: [self.imageid_path_dict.get(x) for x in x]
        )

        # make sure that the data quality frames are sorted as meta
        self.df_ot = self.df_ot.set_index("md5hash")
        self.df_ot = self.df_ot.reindex(index=self.meta_data["md5hash"]).reset_index()

        self.df_lb = self.df_lb.set_index("md5hash")
        self.df_lb = self.df_lb.reindex(index=self.meta_data["md5hash"]).reset_index()

        sorting_hashes = list(self.meta_data["md5hash"].values)
        self.nd_matrix = CleanPatrick.create_annotation_matrix(
            self.df_nd,
            sorting_hashes=sorting_hashes,
        )

        # check the matrix
        # assert df_label_matrix.equals(df_label_matrix.T)
        # CleanPatrick.check_unannotated_pairs(self.df_nd, df_label_matrix)
        # CleanPatrick.check_annotated_pairs(self.df_nd, df_label_matrix)

    @staticmethod
    def check_annotated_pairs(df, df_label_matrix):
        success = True
        for _, row in df.iterrows():
            samples = row["md5hash_list"]
            expected_label = row["label"]
            # Get all unique pairs (order doesn't matter)
            for sample_i, sample_j in itertools.combinations(samples, 2):
                matrix_value = df_label_matrix.loc[sample_i, sample_j]
                if matrix_value != expected_label:
                    print(
                        f"Mismatch for pair ({sample_i}, {sample_j}): expected {expected_label}, got {matrix_value}"
                    )
                    success = False
                # else:
                # print(f"Pair ({sample_i}, {sample_j}) correctly has label {expected_label}.")
        if success:
            print("All annotated pairs are correct.")

    @staticmethod
    def check_unannotated_pairs(df, df_label_matrix):
        annotated_pairs = set()
        for _, row in df.iterrows():
            samples = sorted(row["md5hash_list"])
            for pair in itertools.combinations(samples, 2):
                annotated_pairs.add(pair)

        # Check for every pair in the matrix (only upper triangle for uniqueness)
        all_samples = df_label_matrix.index.tolist()
        success = True
        for i in range(len(all_samples)):
            for j in range(i + 1, len(all_samples)):
                pair = (all_samples[i], all_samples[j])
                if pair not in annotated_pairs:
                    matrix_value = df_label_matrix.loc[all_samples[i], all_samples[j]]
                    if not (pd.isna(matrix_value) or matrix_value is np.nan):
                        print(
                            f"Unexpected non-NaN for unannotated pair {pair}: found {matrix_value}"
                        )
                        success = False
        if success:
            print("All unannotated pairs correctly remain as NaN.")

    @staticmethod
    def create_annotation_matrix(df, sorting_hashes: Optional[list] = None):
        all_samples = set()
        for sample_list in df["md5hash_list"]:
            all_samples.update(sample_list)
        all_samples = sorted(all_samples)
        if sorting_hashes is not None:
            assert len(all_samples) == len(set(sorting_hashes))
            all_samples = sorting_hashes
        n = len(all_samples)

        # Create a mapping from sample name to its matrix index
        sample_to_idx = {sample: idx for idx, sample in enumerate(all_samples)}

        # --- Step 2: Aggregate indices and labels from annotated groups ---
        row_indices = []
        col_indices = []
        label_values = []

        # For every row in the annotation DataFrame, get all unique pair combinations
        for _, row in df.iterrows():
            group_samples = row["md5hash_list"]
            label = row["label"]
            # Convert sample names into indices
            indices = [sample_to_idx[s] for s in group_samples]
            # Iterate over all unique pairs in the group
            for i, j in itertools.combinations(indices, 2):
                # Record both (i, j) and (j, i) for symmetry
                row_indices.extend([i, j])
                col_indices.extend([j, i])
                label_values.extend([label, label])

        # --- Step 3: Create a NumPy array and perform vectorized assignment ---
        # Initialize the matrix with np.nan (meaning "not annotated")
        matrix = np.full((n, n), np.nan)
        # Perform vectorized assignment to fill in the labels for annotated pairs
        matrix[row_indices, col_indices] = label_values
        # Optionally: leave the diagonal as np.nan or set to another value.
        # For example, leave self-comparisons as np.nan:
        np.fill_diagonal(matrix, np.nan)

        # --- Step 4: Convert the NumPy array to a DataFrame ---
        df_label_matrix = pd.DataFrame(matrix, index=all_samples, columns=all_samples)

        return df_label_matrix

    def __getitem__(self, idx):
        ret_values = super().__getitem__(idx=idx)
        if not self.return_meta:
            return ret_values
        img_path = self.meta_data.loc[self.meta_data.index[idx], self.IMG_COL]
        # get the DQI annotations for the image
        sel_df_ot = self.df_ot[self.df_ot["path"] == img_path]
        sel_df_lb = self.df_lb[self.df_lb["path"] == img_path]
        sel_df_nd = self.df_nd[
            self.df_nd["path_list"].apply(lambda lst: img_path in lst)
        ]
        dqi_dict = {
            "off_topic": sel_df_ot["label"].iloc[0],
            "label_error": sel_df_lb["label"].iloc[0],
            "near_duplicate_annos": (
                list(zip(sel_df_nd["path_list"].values, sel_df_nd["label"]))
                if len(sel_df_nd) > 0
                else np.nan
            ),
        }
        return ret_values + (dqi_dict,)

    def get_off_topic_samples(self):
        issue_dict = {k: v for k, v in self.df_ot[["path", "label"]].values}
        return issue_dict

    def get_near_duplicates(self):
        issue_dict = {k: v for k, v in self.df_nd[["path_list", "label"]].values}
        return issue_dict

    def get_label_errors(self):
        issue_dict = {k: v for k, v in self.df_lb[["path", "label"]].values}
        return issue_dict
