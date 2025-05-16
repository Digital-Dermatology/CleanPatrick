from abc import ABC, abstractmethod
from itertools import combinations
from multiprocessing import Pool, RawArray

import numpy as np
import torch
from tqdm import tqdm


class BaseDetector(ABC):

    # A global dictionary storing the variables passed from the initializer.
    var_dict = {}

    @classmethod
    def init_worker(cls, X, dist_shape: tuple, hash_size: int, dataset):
        cls.var_dict["dist_matrix"] = X
        cls.var_dict["dist_shape"] = dist_shape
        cls.var_dict["hash_size"] = hash_size
        cls.var_dict["dataset"] = dataset

    @staticmethod
    @abstractmethod
    def worker_func(row_index: int):
        raise NotImplementedError

    @classmethod
    def get_distance_matrix(
        cls,
        dataset: torch.utils.data.Dataset,
        hash_size: int = 8,
        n_processes: int = 24,
    ):
        dist_shape = (len(dataset), len(dataset))
        dist_mat = np.ones(dist_shape)
        # create the array for the workers
        X = RawArray("d", dist_shape[0] * dist_shape[1])
        X_np = np.frombuffer(X).reshape(dist_shape)
        np.copyto(X_np, dist_mat)
        # start the processing
        with Pool(
            processes=n_processes,
            initializer=cls.init_worker,
            initargs=(X, dist_shape, hash_size, dataset),
        ) as pool:
            list(
                tqdm(
                    pool.imap(cls.worker_func, range(dist_shape[0])),
                    total=dist_shape[0],
                )
            )
        # get the computed matrix from all the workers
        dist_mat = np.frombuffer(X).reshape(dist_shape)

        # copy the upper triangle to the lower triangle
        _dist_mat = dist_mat.copy()
        i_lower = np.tril_indices(_dist_mat.shape[0], -1)
        _dist_mat[i_lower] = _dist_mat.T[i_lower]
        return _dist_mat

    @classmethod
    def get_ranking(
        cls,
        dataset: torch.utils.data.Dataset,
        hash_size: int = 8,
        n_processes: int = 24,
        **kwargs,
    ):
        dist_mat = cls.get_distance_matrix(
            dataset=dataset,
            hash_size=hash_size,
            n_processes=n_processes,
        )
        indexes = list(range(len(dataset)))
        ranking = [(dist_mat[i, j], i, j) for i, j in list(combinations(indexes, 2))]
        ranking = sorted(ranking, key=lambda tup: tup[0], reverse=False)
        ranking = np.asarray(ranking)[:, 1:].astype(int)
        return ranking
