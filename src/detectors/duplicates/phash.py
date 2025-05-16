import imagehash
import numpy as np

from src.detectors.duplicates.duplicate_base import BaseDetector


class pHASHDetector(BaseDetector):
    @staticmethod
    def worker_func(i: int):
        hash_size = pHASHDetector.var_dict["hash_size"]
        dist_matrix = pHASHDetector.var_dict["dist_matrix"]
        dist_shape = pHASHDetector.var_dict["dist_shape"]
        dataset = pHASHDetector.var_dict["dataset"]

        X_np = np.frombuffer(dist_matrix).reshape(dist_shape)
        # compute the hashing in the workers
        img_i = dataset[i][0].convert("L").resize((hash_size + 1, hash_size))
        img_i_hash = imagehash.phash(img_i, hash_size).hash.flatten()
        for j in range(len(dataset)):
            if j >= i:
                img_j = dataset[j][0].convert("L").resize((hash_size + 1, hash_size))
                img_j_hash = imagehash.phash(img_j, hash_size).hash.flatten()
                hamming = img_i_hash.astype(int) - img_j_hash.astype(int)
                dist_hamming = len(hamming[hamming != 0]) / len(hamming)
                X_np[i, j] = dist_hamming
