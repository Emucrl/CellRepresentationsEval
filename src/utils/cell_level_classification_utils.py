import pickle
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from skimage import io
from sklearn.linear_model import LogisticRegression

from src.utils import classification_results_utils, masking_utils, utils


def get_tls_gt(patient: str, cfg: DictConfig) -> pd.Series:
    """Apply the binary mask in cfg.paths.dataset.tls_masks
    with a filename that starts with the patient id

    The cells csv should have x and y values. The function queries the binary mask
    with each cells coordinates

    Args:
        patient (str): patient id (PATIENTID_cells.csv and PATIENTID_mask.png)
        cfg (DictConfig)

    Returns:
        pd.Series: binary mask value for each cells
    """
    cells_csv = pd.read_csv(
        [
            file
            for file in Path(cfg.paths.dataset.cells).rglob("*")
            if file.name == f"{patient}_cells.csv" and file.is_file()
        ][0]
    )
    tls_mask_name = [
        file
        for file in Path(cfg.paths.dataset.tls_masks).rglob("*")
        if file.is_file() and file.name.startswith(patient)
    ][0]
    return masking_utils.apply_mask_cells_df(
        cells_csv,
        io.imread(tls_mask_name),
    )


def get_tumor_gt(patient: str, cfg: DictConfig) -> np.ndarray:
    """Get the tumor gt from cells csv
    Cells csv should have a tumour_mask column
    modify the get_is_tumor function accordingly otherwise

    Args:
        patient (str): patient id (PATIENTID_cells.csv and PATIENTID_mask.png)
        cfg (DictConfig)
    """
    cells_csv = pd.read_csv(
        [
            file
            for file in Path(cfg.paths.dataset.cells).rglob("*")
            if file.name == f"{patient}_cells.csv" and file.is_file()
        ][0]
    )
    cells_csv = utils.get_is_tumor(cells_csv)
    return cells_csv["is_tumour"].values


def get_rawfeatures(patient: str, cfg: DictConfig) -> np.ndarray:
    """Get the dgi features from cells csv
    Args:
        patient (str): patient id (PATIENTID_cells.csv)
        cfg (DictConfig)
    """
    cells_csv = pd.read_csv(
        [
            file
            for file in Path(cfg.paths.dataset.cells).rglob("*")
            if file.name == f"{patient}_cells.csv" and file.is_file()
        ][0]
    )
    return cells_csv[cfg.dgi.training_params.features_list].values


def get_embeddings(patient: str, cfg: DictConfig) -> np.ndarray:
    """Get the dgi embeddings for the patient id
    Args:
        patient (str): patient id (PATIENTID_embeddings.p)
        cfg (DictConfig)
    """
    with open(
        [
            file
            for file in Path(cfg.paths.dataset.embeddings).rglob("*")
            if file.name == f"{patient}_embeddings.p" and file.is_file()
        ][0],
        "rb",
    ) as file:
        embeddings = pickle.load(file)
    return embeddings


def get_normalized_utag(patient: str, cfg: DictConfig) -> np.ndarray:
    """Get the message_passing for the patient id
    Args:
        patient (str): patient id (PATIENTID_message_passing.p)
        cfg (DictConfig)
    """
    with open(
        [
            file
            for file in (Path(cfg.paths.dataset.message_passing)).rglob("*")
            if file.name == f"{patient}_message_passing.p" and file.is_file()
        ][0],
        "rb",
    ) as file:
        embeddings = pickle.load(file)
    return embeddings


def get_dataset(
    patients: List[str],
    cfg: DictConfig,
    data_load_function: Callable,
    gt_load_function: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load cells representations from data_load_function
    and cell gt for all cells with gt_load_function
    for all patients

    Args:
        patients (List[str]): List of patients to use for the dataset
        cfg (DictConfig)
        data_load_function (Callable): one of get_normalized_utag, get_embeddings, get_raw_features
        gt_load_function (Callable): get_tls_gt or get_tumor_gt
    """
    inputs = np.concatenate([data_load_function(patient, cfg) for patient in patients])
    outputs = np.concatenate([gt_load_function(patient, cfg) for patient in patients])
    return inputs, outputs


def classification(
    cfg: DictConfig,
    training_patients: List[str],
    test_patients: List[str],
    data_load_function: Callable,
    gt_load_function: Callable,
) -> Dict:
    """Trains a logistic regression on training_patients patients
    and runs inference on test_patients patients
    Compute binary classifications scores for test set
    Args:
        cfg (DictConfig)
        training_patients (List[str])
        test_patients (List[str])
        data_load_function (Callable)
        gt_load_function (Callable)

    Returns:
        Dict: Binary classification results
    """
    training_inputs, training_outputs = get_dataset(
        training_patients, cfg, data_load_function, gt_load_function
    )
    test_inputs, test_outputs = get_dataset(
        test_patients, cfg, data_load_function, gt_load_function
    )
    model = LogisticRegression(
        class_weight="balanced",
    )
    model = model.fit(training_inputs, training_outputs)
    probas = model.predict_proba(test_inputs)
    results = classification_results_utils.get_binary_classif_results(
        torch.tensor(probas, dtype=torch.float), torch.tensor(test_outputs)
    )
    results["exp"] = f"{data_load_function.__name__.split('_')[1]}"
    results["probas"] = probas[:, 1]
    results["labels"] = test_outputs.astype(float)
    return {key: [value] for key, value in results.items()}


def classifications(
    config: DictConfig,
    training_patients: List[str],
    test_patients: List[str],
    gt_load_function: Callable,
) -> pd.DataFrame:
    """Runs binary classification logreg for all 3 cell representations
    (raw features, dgi embeddings, message passing embeddings)
    """
    results = []

    result = classification(
        config,
        training_patients=training_patients,
        test_patients=test_patients,
        data_load_function=get_rawfeatures,
        gt_load_function=gt_load_function,
    )

    results += [result]

    result = classification(
        config,
        training_patients=training_patients,
        test_patients=test_patients,
        data_load_function=get_embeddings,
        gt_load_function=gt_load_function,
    )
    results += [result]

    result = classification(
        config,
        training_patients=training_patients,
        test_patients=test_patients,
        data_load_function=get_normalized_utag,
        gt_load_function=gt_load_function,
    )
    results += [result]

    results_df = pd.concat([pd.DataFrame(el) for el in results], ignore_index=True)
    return results_df
