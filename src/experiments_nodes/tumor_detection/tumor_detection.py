"""tls binary classification
"""
import logging
import pickle
import random
import warnings
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from src.experiments_nodes.tumor_detection import tumor_gcn_baseline
from src.utils import cell_level_classification_utils, gcn_classification_utils

warnings.simplefilter(action="ignore", category=FutureWarning)
log = logging.getLogger(__name__)


def tumor_classification_pipeline(
    config: DictConfig, output_folder: Path, iteration: int = 0
) -> None:
    """One iteration of the tumor classification pipeline

    Linear classifiers are evaluated on raw_features/dgi embeddings/message passing embeddings
    A supervised GCN is also evaluated

    Cells files should have a tumour_mask column

    Multiple training dataset sizes are used to demonstrate the ability to learn from small datasets
    Args:
        config (DictConfig)
        output_folder (Path): Folder to save results
    """
    results = []

    patients = [
        filename.stem.replace("_embeddings", "")
        for filename in Path(config.paths.dataset.embeddings).rglob("*")
        if filename.is_file()
    ]
    random.shuffle(patients)

    test_patients = pickle.load(
        open(output_folder / f"splits/test_patients_{iteration}.pkl", "rb")
    )
    training_patients = [
        patient for patient in patients if patient not in test_patients
    ]
    if len(test_patients) < 1:
        log.info("No test patients")
        return

    log.info(
        "%s training patients %s test_patients",
        len(training_patients),
        len(test_patients),
    )
    for train_frac in [0.05, 0.1, 0.15, 0.20, 0.3, 0.5, 0.75, 1]:
        log.info("Train ratio %s", train_frac)
        sampled_training_patients = training_patients[
            : int(train_frac * len(training_patients))
        ]
        log.info(
            "%s training, %s test",
            len(sampled_training_patients),
            len(test_patients),
        )
        if len(sampled_training_patients) < 1 or len(test_patients) < 1:
            continue
        linear_result = cell_level_classification_utils.classifications(
            config,
            sampled_training_patients,
            test_patients,
            cell_level_classification_utils.get_tumor_gt,
        )
        log.info("Supervised GCN")
        gcn_result = gcn_classification_utils.node_level_binary_gcn(
            training_patients, test_patients, config, tumor_gcn_baseline.GraphNodeLabel
        )
        gcn_result["exp"] = "supervised gcn"

        ratio_result = linear_result.append(gcn_result, ignore_index=True)

        ratio_result["training_ratio"] = train_frac
        results += [ratio_result]

    output_folder.mkdir(parents=True, exist_ok=True)
    results_df = pd.concat(results, ignore_index=True)
    results_df["iteration"] = iteration
    results_df.to_parquet(
        output_folder / "classification_results.parquet",
        partition_cols=["iteration", "exp", "training_ratio"],
    )
