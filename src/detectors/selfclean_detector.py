from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.core.src.pkg import Embedder, embed_dataset
from src.detectors.selfclean.selfclean.cleaner.selfclean_cleaner import SelfCleanCleaner


class SelfCleanDetector:
    @classmethod
    def get_ranking(
        cls,
        dataset: torch.utils.data.Dataset,
        ckp_path: str,
        emb_space: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if emb_space is None and labels is None:
            model, _, _ = Embedder.load_dino(
                ckp_path=ckp_path,
                return_info=True,
                n_head_layers=0,
            )
            model = model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            torch_dataset = DataLoader(
                dataset,
                batch_size=128,
                drop_last=False,
                shuffle=False,
                collate_fn=(
                    dataset.__class__.collate_fn
                    if hasattr(dataset.__class__, "collate_fn")
                    else None
                ),
            )
            emb_space, labels, _, _ = embed_dataset(
                torch_dataset=torch_dataset,
                model=model,
                n_layers=1,
                memmap=False,
                normalize=True,
            )
            del _

        issues = SelfCleanDetector.get_ranking_from_emb_space(
            emb_space=emb_space,
            labels=labels,
            dataset=dataset,
        )
        return issues

    @classmethod
    def get_ranking_from_emb_space(
        cls,
        emb_space: np.ndarray,
        labels: np.ndarray,
        dataset: torch.utils.data.Dataset,
        **kwargs,
    ):
        cleaner = SelfCleanCleaner(
            chunk_size=512,
            memmap=False,
            global_leaves=False,
            auto_cleaning=False,
        )
        cleaner = cleaner.fit(
            emb_space=emb_space,
            labels=labels,
            class_labels=dataset.classes,
        )
        issues = cleaner.predict()
        issues.issue_dict["near_duplicate_matrix"] = cleaner.distance_matrix
        return issues
