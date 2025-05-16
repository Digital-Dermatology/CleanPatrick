import numpy as np
from skimage.metrics import structural_similarity as ssim

from src.detectors.duplicates.duplicate_base import BaseDetector


class SSIMDetector(BaseDetector):
    @staticmethod
    def worker_func(i: int):
        hash_size = SSIMDetector.var_dict["hash_size"]
        dist_matrix = SSIMDetector.var_dict["dist_matrix"]
        dist_shape = SSIMDetector.var_dict["dist_shape"]
        dataset = SSIMDetector.var_dict["dataset"]

        X_np = np.frombuffer(dist_matrix).reshape(dist_shape)
        # compute the hashing in the workers
        img_i = dataset[i][0].convert("L").resize((hash_size + 1, hash_size))
        for j in range(len(dataset)):
            if j >= i:
                img_j = dataset[j][0].convert("L").resize((hash_size + 1, hash_size))
                dist = 1 - ssim(
                    np.asarray(img_i),
                    np.asarray(img_j),
                )
                X_np[i, j] = dist
