import copy
from pathlib import Path

import numpy as np
from loguru import logger
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from torchvision import transforms

from src.core.src.utils.plotting import calculate_scores_from_ranking
from src.dataset.cleanpatrick import CleanPatrick
from src.detectors.label_errors.confident_learning import ConfidentLearningDetector
from src.detectors.label_errors.noise_rank import NoiseRankDetector
from src.detectors.off_topic_samples.pyod_wrapper import PyODWrapper
from src.detectors.selfclean_detector import SelfCleanDetector


def get_sorted_labels(
    label_matrix: np.ndarray, distance_matrix: np.ndarray
) -> np.ndarray:
    """
    Given a label and distance matrix, returns sorted labels based on ascending distances.
    """
    tril_indices = np.tril_indices_from(label_matrix, k=-1)
    labels = label_matrix[tril_indices]
    distances = distance_matrix[tril_indices]
    mask = ~np.isnan(labels)
    return labels[mask][np.argsort(distances[mask])]


def evaluate_off_topic(dataset):
    """
    Evaluate off-topic sample detection for various methods.
    Returns a dict mapping method names to their truth arrays.
    """
    # Ground truth indices
    ot_samples = set(dataset.df_ot.loc[dataset.df_ot["label"] == 1].index)
    total = len(dataset)
    pct = len(ot_samples) / total * 100
    logger.info(
        f"Off-Topic: {len(ot_samples)}/{total} ({pct:.2f}%) samples are off-topic"
    )

    results = {}
    # SelfClean
    issues = SelfCleanDetector.get_ranking(
        dataset=dataset, ckp_path="models/Fitzpatrick17k/DINO/checkpoint-epoch500.pth"
    )
    preds = np.array(
        [
            1 if idx in ot_samples else 0
            for idx in issues["off_topic_samples"]["indices"]
        ]
    )
    results["SelfClean"] = preds

    # PyOD methods
    for name, model in [
        ("ECOD", ECOD),
        ("IForest", IForest),
        ("KNN", KNN),
        ("HBOS", HBOS),
    ]:
        ranking = PyODWrapper.get_ranking(irrelevant_detector=model, dataset=dataset)
        preds = np.array([1 if idx in ot_samples else 0 for idx in ranking])
        results[name] = preds

    _print_scores(results)
    return results


def evaluate_near_duplicates(dataset):
    """
    Evaluate near-duplicate detection for various methods.
    """
    # SelfClean distances
    issues = SelfCleanDetector.get_ranking(
        dataset=dataset, ckp_path="models/Fitzpatrick17k/DINO/checkpoint-epoch500.pth"
    )
    nd_self = get_sorted_labels(
        dataset.nd_matrix.values, issues["near_duplicate_matrix"]
    )

    dataset.transform = None  # disable transforms for hash/ssim
    results = {"SelfClean": nd_self}
    for method in ["phash", "ssim"]:
        dist_mat = np.load(f"assets/results/{method}_matrix.npy")
        nd_vals = get_sorted_labels(dataset.nd_matrix.values, dist_mat)
        results[method.upper()] = nd_vals
    dataset.transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    _print_scores(results)
    return results


def evaluate_label_errors(dataset):
    """
    Evaluate label error detection for various methods.
    """
    lbl_errs = set(dataset.df_lb.loc[dataset.df_lb["label"] == 1].index)
    total = len(dataset)
    pct = len(lbl_errs) / total * 100
    logger.info(
        f"Label Errors: {len(lbl_errs)}/{total} ({pct:.2f}%) samples are mislabeled"
    )

    results = {}
    # SelfClean
    issues = SelfCleanDetector.get_ranking(
        dataset=dataset, ckp_path="models/Fitzpatrick17k/DINO/checkpoint-epoch500.pth"
    )
    preds = np.array(
        [1 if idx in lbl_errs else 0 for idx in issues["label_errors"]["indices"]]
    )
    results["SelfClean"] = preds

    # Other detectors
    for Detector in [ConfidentLearningDetector, NoiseRankDetector]:
        ranking = Detector.get_ranking(dataset=copy.deepcopy(dataset))
        ids = [item[1] if isinstance(item, tuple) else item for item in ranking]
        preds = np.array([1 if idx in lbl_errs else 0 for idx in ids])
        results[Detector.__name__] = preds

    _print_scores(results)
    return results


def _print_scores(results: dict):
    """
    Helper to print scores for each method using calculate_scores_from_ranking.
    """
    for name, truth in results.items():
        logger.info(f"Method: {name}")
        calculate_scores_from_ranking(
            ranking=truth, log_wandb=False, show_plots=False, show_scores=True
        )
    logger.info("-" * 50)


def main():
    # Setup
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    data_path = Path("data/")
    dataset = CleanPatrick(
        cleanpatrick_dataset_dir=data_path / "CleanPatrick",
        fitzpatrick_dataset_dir=data_path / "fitzpatrick17k",
        fitzpatrick_csv_file=data_path / "fitzpatrick17k/fitzpatrick17k.csv",
        transform=transform,
        return_path=True,
    )

    logger.info("Starting Benchmark Evaluation")
    evaluate_off_topic(dataset)
    evaluate_near_duplicates(dataset)
    evaluate_label_errors(dataset)
    logger.info("Benchmark Evaluation Complete")


if __name__ == "__main__":
    main()
