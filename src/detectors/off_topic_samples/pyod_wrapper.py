from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.core.src.pkg import Embedder, embed_dataset


class PyODWrapper:
    @classmethod
    def get_ranking(
        cls,
        irrelevant_detector: Callable,
        dataset: torch.utils.data.Dataset,
        emb_space: Optional[np.ndarray] = None,
        ssl_model: str = "imagenet_vit_tiny",
        batch_size: int = 16,
        n_layers: int = 1,
        **kwargs,
    ):
        if emb_space is None:
            model = Embedder.load_pretrained(ssl_model, n_head_layers=0)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            torch_dataset = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
            )

            emb_space, _, _, _ = embed_dataset(
                torch_dataset=torch_dataset,
                model=model,
                n_layers=n_layers,
                memmap=False,
            )

        ranking = PyODWrapper.get_ranking_from_emb_space(
            irrelevant_detector=irrelevant_detector,
            emb_space=emb_space,
        )
        return ranking

    @classmethod
    def get_ranking_from_emb_space(
        cls,
        irrelevant_detector: Callable,
        emb_space: np.ndarray,
        **kwargs,
    ):
        clf = irrelevant_detector()
        clf.fit(emb_space)
        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores
        ranking = np.argsort(-y_train_scores)
        return ranking
