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


def get_sorted_labels(label_matrix, distance_matrix):
    # Get the lower triangular indices (excluding the diagonal).
    tril_indices = np.tril_indices_from(label_matrix, k=-1)
    # Extract the lower triangular portions from both matrices.
    labels_lower = label_matrix[tril_indices]
    distances_lower = distance_matrix[tril_indices]
    # Filter out entries where the label is np.inf (i.e. non-annotated pairs).
    valid_mask = ~np.isnan(labels_lower)
    filtered_labels = labels_lower[valid_mask]
    filtered_distances = distances_lower[valid_mask]
    # Get the indices that would sort the distances in ascending order.
    sorted_order = np.argsort(filtered_distances)
    # Reorder the labels by the sorted order of their corresponding distances.
    sorted_labels = filtered_labels[sorted_order]
    return sorted_labels


if __name__ == "__main__":
    # load the dataset to evaluate on
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    data_path = Path("data/")
    dataset_path = data_path / "fitzpatrick17k/"
    csv_path = dataset_path / "fitzpatrick17k.csv"
    dataset = CleanPatrick(
        cleanpatrick_dataset_dir=data_path / "CleanPatrick",
        fitzpatrick_dataset_dir=dataset_path,
        fitzpatrick_csv_file=csv_path,
        transform=transform,
        return_path=True,
    )

    # run SelfClean since it can detect all issue types
    issues = SelfCleanDetector.get_ranking(
        dataset=dataset,
        ckp_path="models/Fitzpatrick17k/DINO/checkpoint-epoch500.pth",
    )
    pred_selfclean_ot_indices = issues["off_topic_samples"]["indices"]
    pred_selfclean_lbl_indices = issues["label_errors"]["indices"]
    pred_selfclean_nd_matrix = issues["near_duplicate_matrix"]

    # --- Off-Topic Samples ---
    # create ground truth metadata
    ot_samples = list(dataset.df_ot[dataset.df_ot["label"] == 1].index)
    # calculate the % of off-topic samples in the dataset
    print(f"% of off-topic samples: {(len(ot_samples) / len(dataset)) * 100}")

    l_truths = []
    # eval SelfClean
    truth = [1 if int(x) in ot_samples else 0 for x in pred_selfclean_ot_indices]
    truth = np.asarray(truth)
    l_truths.append(("SelfClean", truth))

    for name_detector, irr_detector in [
        ("ECOD", ECOD),
        ("IForest", IForest),
        ("KNN", KNN),
        ("HBOS", HBOS),
    ]:
        ranking = PyODWrapper.get_ranking(
            irrelevant_detector=irr_detector,
            dataset=dataset,
        )
        ranking_target = [1 if x in ot_samples else 0 for x in ranking]
        ranking_target = np.asarray(ranking_target)
        l_truths.append((name_detector, ranking_target))

    for name, truth in l_truths:
        logger.info(f"Method: {name.upper()}")
        calculate_scores_from_ranking(
            ranking=truth,
            log_wandb=False,
            show_plots=False,
            show_scores=True,
        )
    logger.info("-" * 40)
    # -----------------------

    # --- Near Duplicates ---
    l_truths = []
    nd_truth = get_sorted_labels(dataset.nd_matrix.values, pred_selfclean_nd_matrix)
    nd_truth = np.asarray(nd_truth)
    l_truths.append(("SelfClean", nd_truth))
    # eval competitors
    dataset.transform = None
    for dup_detector in ["phash", "ssim"]:
        dist_mat = np.load(f"assets/results/{dup_detector}_matrix.npy")
        nd_truth = get_sorted_labels(dataset.nd_matrix.values, dist_mat)
        nd_truth = np.asarray(nd_truth)
        l_truths.append((dup_detector, nd_truth))
    dataset.transform = transform
    for name, truth in l_truths:
        logger.info(f"Method: {name.upper()}")
        calculate_scores_from_ranking(
            ranking=truth,
            log_wandb=False,
            show_plots=False,
            show_scores=True,
        )
    logger.info("-" * 40)
    # -----------------------

    # --- Label Errors ---
    # create ground truth metadata
    lbl_errs = list(dataset.df_lb[dataset.df_lb["label"] == 1].index)
    # calculate the % of label errors in the dataset
    print(f"% of label errors: {(len(lbl_errs) / len(dataset)) * 100}")

    l_truths = []
    # eval SelfClean
    truth = [1 if int(x) in lbl_errs else 0 for x in pred_selfclean_lbl_indices]
    truth = np.asarray(truth)
    l_truths.append(("SelfClean", truth))

    # eval competitors
    for lbl_detector in [
        ConfidentLearningDetector,
        NoiseRankDetector,
    ]:
        ranking = lbl_detector.get_ranking(
            dataset=copy.deepcopy(dataset),
        )
        if type(ranking[0]) is tuple:
            truth = [1 if int(x[1]) in lbl_errs else 0 for x in ranking]
        else:
            truth = [1 if int(x) in lbl_errs else 0 for x in ranking]
        truth = np.asarray(truth)
        l_truths.append((str(lbl_detector), truth))

    for name, truth in l_truths:
        logger.info(f"Method: {name.upper()}")
        calculate_scores_from_ranking(
            ranking=truth,
            log_wandb=False,
            show_plots=False,
            show_scores=True,
        )
    # -----------------------
