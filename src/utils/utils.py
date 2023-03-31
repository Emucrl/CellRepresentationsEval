"""miscellaneous utils"""
import importlib
import pathlib
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def get_is_tumor(cells_csv: pd.DataFrame) -> pd.DataFrame:
    """look for cell_level ground truth for tumor/non_tumor in the cells_csv
    To be adapted for different datasets
    Args:
        cells_csv (pd.DataFrame)

    Returns:
        pd.DataFrame: cells_csv with an additional "is_tumor" column
    """
    if "probabilities.tumor" in cells_csv.columns:
        cells_csv["is_tumour"] = ~(cells_csv["probabilities.tumor"] < 0.5) | (
            cells_csv["celltypes"] == "Tumor"
        )
    else:
        cells_csv["is_tumour"] = ~(
            cells_csv["tumour_mask"].str.lower() == "non_tumour"
        ) | (cells_csv["celltypes"] == "Tumor")
    return cells_csv


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.

    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    if obj_path == "":
        obj_path = "builtins"
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object '{obj_name}' cannot be loaded from '{obj_path}'.")
    return getattr(module_obj, obj_name)


def swap_root_directory(
    source_directory: Path, target_directory: Path, file: Path
) -> Path:
    """replace source_directory by target_directory in file"""
    index = file.parts.index(source_directory.parts[-1])
    new_path = pathlib.Path(target_directory).joinpath(*file.parts[index + 1 :])
    return new_path


def generate_test_sets_tls(
    config: DictConfig, output_path: Path, n_iterations: int
) -> None:
    """train/test splits only on rois with tls masks"""
    tls_masks = [
        el for el in Path(config.paths.dataset.tls_masks).rglob("*") if el.is_file()
    ]
    patients_with_tls = [patient.stem.replace("_mask", "") for patient in tls_masks]
    (output_path / "splits").mkdir(parents=True, exist_ok=True)
    for iteration in range(n_iterations):
        test_patients = np.random.choice(
            patients_with_tls, int(0.25 * len(patients_with_tls))
        )
        with open(output_path / f"splits/test_patients_{iteration}.pkl", "wb") as file:
            pickle.dump(test_patients, file)


def generate_test_sets(
    config: DictConfig, output_path: Path, n_iterations: int
) -> None:
    """generate n random test sets with random selection"""
    patients = [
        filename.stem.replace("_embeddings", "")
        for filename in Path(config.paths.dataset.embeddings).rglob("*")
        if filename.is_file()
    ]
    (output_path / "splits").mkdir(parents=True, exist_ok=True)
    for iteration in range(n_iterations):
        test_patients = list(np.random.choice(patients, int(0.25 * len(patients))))
        with open(output_path / f"splits/test_patients_{iteration}.pkl", "wb") as file:
            pickle.dump(test_patients, file)


def find_patient_clinical(config: DictConfig) -> pd.DataFrame:
    """load clinical info dataframe, keep rows for which embeddings have been generated
    and binarize the infiltration

    Returns:
        pd.DataFrame: should contains at least:
            - patient column with patient id
            - indication column with the corresponding binarized infiltration label
    """
    clinical = pd.read_parquet(config.paths.clinical_info_df)

    patients = [
        file.name.split("_")[0]
        for file in Path(config.paths.dataset.embeddings).rglob("*")
        if file.is_file()
    ]
    clinical = clinical[clinical["patient"].isin(patients)].rename(
        columns={"Lymphocyte infiltration": "infiltration"}
    )

    clinical = clinical[~clinical["infiltration"].isin(["0", "Not applicable"])]
    clinical["infiltration"] = (clinical["infiltration"] == "3").astype(int)

    return clinical


def generate_test_sets_by_col(
    config: DictConfig, output_path: Path, n_iterations: int, col: str
) -> None:
    """generate test set with stratified sampling on col"""
    patients_df = find_patient_clinical(config)
    if col == "infiltration":
        patients_df = patients_df[
            ~patients_df["infiltration"].isin(["0", "Not applicable"])
        ]

    (output_path / "splits").mkdir(parents=True, exist_ok=True)
    for iteration in range(n_iterations):
        test_patients = list(
            patients_df.groupby(col, group_keys=False)
            .apply(lambda x: x.sample(frac=0.25))["patient"]
            .values
        )
        with open(output_path / f"splits/test_patients_{iteration}.pkl", "wb") as file:
            pickle.dump(test_patients, file)


def find_patient_indications(cfg: DictConfig) -> pd.DataFrame:
    """Parse filenames to find the groundtruth indication

    Returns:
        pd.DataFrame: should contains at least:
            - patient column with patient roi id
            - indication column with the corresponding indication label
    """
    files = [
        file for file in Path(cfg.paths.dataset.embeddings).rglob("*") if file.is_file()
    ]
    indications = [
        re.search(  # type: ignore
            r"(?<=indication=)(.*)(?=/datadump=)", file.as_posix()
        ).group(1)
        for file in files
    ]
    datadumps = [
        re.search(r"(?<=/datadump=)(.*)(?=/)", file.as_posix()).group(1)  # type: ignore
        for file in files
    ]
    patients = ["_".join(file.name.split("_")[:-1]) for file in files]
    dataset = pd.DataFrame(
        {"indication": indications, "patient": patients, "datadump": datadumps}
    )
    return dataset[dataset["indication"].isin(["NSCLC", "SCCHN", "BC"])]
