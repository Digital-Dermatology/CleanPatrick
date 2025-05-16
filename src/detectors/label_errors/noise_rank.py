from math import floor
from typing import List, Optional

import faiss
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from src.core.src.pkg import Embedder, embed_dataset


class NoiseRankDetector:
    @classmethod
    def get_ranking(
        cls,
        dataset: torch.utils.data.Dataset,
        emb_space: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        k: int = 5,
        b: int = 1,
        e: int = 1,
        alpha: float = 0.5,
        b_factor: float = 1.0,
        ssl_model: str = "imagenet_vit_tiny",
        batch_size: int = 16,
        n_layers: int = 1,
        seed: int = 42,
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

        # 1. Generate class prototypes
        prototypes = cls.generate_prototypes(
            X=emb_space,
            y=labels,
            num_classes=np.unique(labels).shape[0],
            random_seed=seed,
        )
        y_proto = [[k] * len(v) for k, v in prototypes.items()]
        y_proto = np.array([item for sublist in y_proto for item in sublist])

        # 2. Generating label predictions
        y_proto_prime = cls.weighted_knn_predictions_faiss(
            X=emb_space,
            y=labels,
            prototypes=list(prototypes.values()),
            k=k,
            b=b,
            e=e,
        )
        assert len(y_proto_prime) == len(
            y_proto
        ), "y_proto and y_proto^\prime need to have the same length"

        # 3. Dependence graph construction
        x_proto = np.concatenate(list(prototypes.values()))
        # The rank reflects the relative likelihood of being mislabeled
        # and the impact on mispredictions, accounted for in the penalty function
        noise_rank_score = cls.dependence_graph(
            X=emb_space,
            y=labels,
            x_proto=x_proto,
            y_proto=y_proto,
            y_proto_prime=y_proto_prime,
            b=b,
            e=e,
            alpha=alpha,
            b_factor=b_factor,
        )
        noise_rank_score = sorted(
            enumerate(noise_rank_score), key=lambda tup: tup[-1], reverse=True
        )
        noise_rank_score = [(x[1], x[0]) for x in noise_rank_score]
        return noise_rank_score

    @staticmethod
    def generate_prototypes(
        X: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        random_seed: int = 42,
    ):
        """
        Generate class prototypes using K-means clustering.

        X: array-like, shape (n_samples, n_features) - Input samples.
        y: array-like, shape (n_samples,) - Class labels for input samples.
        num_classes: int - The number of unique classes.
        """
        # Calculate the number of prototypes (cluster centroids)
        class_instances = [sum(y == c) for c in range(num_classes)]
        rho = np.mean(class_instances)
        k = max(floor((rho / 2) ** 0.5), 1)
        # Compute the prototypes
        prototypes = {}
        for c in range(num_classes):
            # Filter instances by class
            class_instances = X[y == c]
            if len(class_instances) == 0:
                continue
            # Clip k if the number of instances is smaller
            _k = min(k, len(class_instances))
            # Perform K-means clustering to find prototypes
            kmeans = KMeans(n_clusters=_k, random_state=random_seed).fit(
                class_instances
            )
            prototypes[c] = kmeans.cluster_centers_
        return prototypes

    @staticmethod
    def weighted_knn_predictions_faiss(
        X: np.ndarray,
        y: np.ndarray,
        prototypes: List[np.ndarray],
        k: int,
        b: float,
        e: float,
    ):
        """
        X: array-like, shape (n_samples, n_features) - Input samples.
        y: array-like, shape (n_samples,) - Labels for input samples.
        prototypes: list, shape [$\sqrt{\rho/2}$] * C - Prototypes.
        k: int - Number of neighbors to use.
        b: float - Bias term in the distance kernel function.
        e: float - Weight exponent in the distance kernel function.
        """

        # Flatten the list of prototypes and create a corresponding label array
        prototype_array = np.vstack(prototypes)

        # Convert data to float32 as required by FAISS
        X = X.astype(np.float32)
        prototype_array = prototype_array.astype(np.float32)

        # Build a FAISS index for the embeddings
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)

        # Find the k nearest neighbors for each prototype
        distances, indices = index.search(prototype_array, k)

        # Calculate weights and make predictions
        predictions = np.zeros(prototype_array.shape[0], dtype=int)

        for i in range(prototype_array.shape[0]):
            # Calculate weights using the kernel function
            weights = 1 / (b + distances[i] ** e)

            # Perform the weighted vote
            vote_dict = {}
            for j, weight in zip(indices[i], weights):
                neighbor_label = y[j]
                vote_dict[neighbor_label] = vote_dict.get(neighbor_label, 0) + weight

            # Predict the label with the highest weighted vote
            predictions[i] = max(vote_dict, key=vote_dict.get)

        return predictions

    @staticmethod
    def dependence_graph(
        X: np.ndarray,
        y: np.ndarray,
        x_proto: np.ndarray,
        y_proto: np.ndarray,
        y_proto_prime: np.ndarray,
        b: float,
        e: float,
        alpha: float,
        b_factor: float,
    ):
        x_i = X[:, np.newaxis, :]
        y_i = y[:, np.newaxis]
        x_j = x_proto[np.newaxis, :]
        y_j = y_proto[np.newaxis, :]
        y_j_prime = y_proto_prime[np.newaxis, :]
        x_dist = np.linalg.norm(x_i - x_j, axis=-1)
        ij = y_i == y_j
        ijprime = y_i == y_j_prime
        jjprime = y_j == y_j_prime
        weight = (
            # c11, is for all pairs of examples (i, j) that share the same given label (y_i = y_j)
            ij * (-1)
            +
            # c10: different labels (i.e., yi \neq yj) where y_j^\prime = y_j
            (~ij & jjprime) * (1 - alpha)
            +
            # c01: different labels ($y_i \neq y_j$) where $y_j^{\prime} \neq y_j$ and $y_j^{\prime} \neq y_i$
            (~ij & ~jjprime & ~ijprime) * alpha
            +
            # c00: different labels ($y_i \neq y_j$) where $y_j^{\prime} \neq y_j$ and $y_j^{\prime}=y_i$.
            (~ij & ~jjprime & ijprime) * alpha * b_factor
        )
        return (weight / (b + x_dist**e)).sum(axis=-1)
