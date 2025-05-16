from typing import Optional

import cleanlab
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import AdaBoostClassifier
from torch.utils.data import DataLoader

from src.core.src.pkg import Embedder, embed_dataset


class ConfidentLearningDetector:
    @classmethod
    def get_ranking(
        cls,
        dataset: torch.utils.data.Dataset,
        emb_space: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        label_classes: Optional[list] = None,
        ssl_model: str = "imagenet_vit_tiny",
        batch_size: int = 16,
        n_layers: int = 1,
        return_scores: bool = False,
        **kwargs,
    ):
        if emb_space is None and labels is None:
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

            emb_space, labels, _, _ = embed_dataset(
                torch_dataset=torch_dataset,
                model=model,
                n_layers=n_layers,
                memmap=False,
            )

        # cleanlab expects array starting from zero in range
        if label_classes is None:
            label_classes = dataset.classes

        ranking = ConfidentLearningDetector.get_ranking_from_emb_space(
            emb_space=emb_space,
            labels=labels,
            label_classes=label_classes,
            return_scores=return_scores,
        )
        return ranking

    @classmethod
    def get_ranking_from_emb_space(
        cls,
        emb_space: np.ndarray,
        labels: np.ndarray,
        label_classes: list,
        cv_n_folds: int = 10,
        return_scores: bool = False,
        **kwargs,
    ):
        # cleanlab expects array starting from zero in range
        label_names = [label_classes[x] for x in labels]
        reindexed_labels = pd.factorize(label_names)[0]

        cl = cleanlab.classification.CleanLearning(
            AdaBoostClassifier(random_state=0),
            cv_n_folds=cv_n_folds,
        )
        label_issues = cl.find_label_issues(emb_space, reindexed_labels)
        ranking = list(label_issues.sort_values(by="label_quality").index)
        if return_scores:
            return np.stack([list(label_issues["label_quality"]), ranking], axis=1)
        return ranking
