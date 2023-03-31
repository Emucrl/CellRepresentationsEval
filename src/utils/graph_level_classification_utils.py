import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.utils import clustering_utils, multiclass_utils


def generate_avg_embeddings(
    cfg: DictConfig, global_representations_folder: Path
) -> None:
    """for all embeddings file in from cfg.paths.dataset.embeddings
    save the file-level avg in the representations_folder
    (1, d) representation for each ROI"""

    embeddings_files = [
        file for file in Path(cfg.paths.dataset.embeddings).rglob("*") if file.is_file()
    ]

    bags_of_embeddings_path = (
        global_representations_folder / f"avg_embeddings_{cfg.exp_name}"
    )
    bags = []
    for filename in tqdm(embeddings_files):
        with open(filename, "rb") as file:
            embedding = pickle.load(file).mean(0)

        patient = "_".join(filename.name.split("_")[:-1])
        bags += [
            {
                "bag_of_embedding": embedding,
                "patient": patient,
            }
        ]

    pd.DataFrame(bags).to_parquet(
        bags_of_embeddings_path,
        partition_cols=["patient"],
    )


def generate_bags_of_raw_features(
    cfg: DictConfig,
    global_representations_folder: Path,
    clustering: KMeans,
    training_patients: Iterable[str],
) -> None:
    """
    Generate bags of raw features for all cells in cfg.paths.dataset.cells
    Cells for patients in training_patients are pooled and clustered.
    Centroids of those clusters are used to apply the clustering to all other patients

    All patients are then represented by the frequency of each clusters in the patient cells

    Args:
        cfg (DictConfig)
        global_representations_folder (Path): Folder to save the patient representations
        clustering (KMeans): Clustering to use to identify the bags
        training_patients (Iterable[str]): list of patients to use compute the bags centroids
    """
    # Apply clustering to centroids_rglob cells
    cells_files = [
        el
        for patient in training_patients
        for el in list(Path(cfg.paths.dataset.cells).rglob(f"*/*/{patient}_cells*"))
    ]

    centroid_cells = pd.concat(pd.read_csv(cells_file) for cells_file in cells_files)[
        cfg.dgi.training_params.features_list
    ]
    centroid_cells.insert(
        column="cluster",
        loc=0,
        value=clustering.fit_predict(centroid_cells.values),
    )

    # Compute centroids from the previous clutering
    centroids = centroid_cells.groupby("cluster").mean()
    recipients_cells_files = list(
        el for el in Path(cfg.paths.dataset.cells).rglob("*") if el.is_file()
    )

    bags = []
    # Apply centroids to all cells
    for recipient in recipients_cells_files:
        recipient_cells = pd.read_csv(recipient)

        recipient_cells.insert(
            loc=0,
            column="cluster",
            value=clustering_utils.apply_centroids(
                recipient_cells[cfg.dgi.training_params.features_list], centroids
            ),
        )
        bag_of_cells = (
            recipient_cells.value_counts("cluster").sort_index()
            / recipient_cells.shape[0]
        )
        bag_of_cells = bag_of_cells.reindex(
            list(range(centroids.shape[0])), fill_value=0
        )

        patient = "_".join(recipient.name.split("_")[:-1])
        bags += [
            {
                "bag_of_embedding": bag_of_cells.values,
                "patient": patient,
            }
        ]
    pd.DataFrame(bags).to_parquet(
        global_representations_folder / "bags_of_raw_features",
        partition_cols=["patient"],
    )


def generate_avg_embeddings_by_patients(
    cfg: DictConfig, global_representations_folder: Path
) -> None:
    """Load embeddings for all ROIs of a single patient
    save the mean in the representations_folder, (1, d) representation for each patient
    """

    embeddings_files = [
        file for file in Path(cfg.paths.dataset.embeddings).rglob("*") if file.is_file()
    ]

    patients = set(file.name.split("_ROI")[0] for file in embeddings_files)

    bags_of_embeddings_path = (
        global_representations_folder / f"avg_embeddings_{cfg.exp_name}"
    )
    bags = []
    for patient in tqdm(patients):
        patient_files = [
            file for file in embeddings_files if file.name.startswith(patient)
        ]
        embedding = np.concatenate(
            [pickle.load(open(filename, "rb")) for filename in patient_files]
        ).mean(0)

        bags += [
            {
                "bag_of_embedding": embedding,
                "patient": patient,
            }
        ]

    pd.DataFrame(bags).to_parquet(
        bags_of_embeddings_path,
        partition_cols=["patient"],
    )


def generate_bags_of_raw_features_by_patients(
    cfg: DictConfig,
    global_representations_folder: Path,
    clustering: KMeans,
    training_patients: Iterable[str],
) -> None:
    """
    Generate bags of raw features for all patients.
    Cells of all ROIs of the same patients are pooled and clustered.
    Centroids of those clusters are used to apply the clustering to all other patients

    All patients are then represented by the frequency of each clusters in the patient cells

    Args:
        cfg (DictConfig)
        global_representations_folder (Path): Folder to save the patient representations
        clustering (KMeans): Clustering to use to identify the bags
        training_patients (Iterable[str]): list of patients to use compute the bags centroids
    """
    # Apply clustering to centroids_rglob cells
    cells_files = [
        el
        for el in list(Path(cfg.paths.dataset.cells).rglob("*/*/*_cells*"))
        if el.is_file()
    ]
    patients = set(file.name.split("_ROI")[0] for file in cells_files)
    centroid_cells = pd.concat(
        [
            pd.concat(
                pd.read_csv(cells_file)
                for cells_file in cells_files
                for patient in training_patients
                if cells_file.name.startswith(patient)
            )
        ],
        ignore_index=True,
    )[cfg.dgi.training_params.features_list]
    centroid_cells.insert(
        column="cluster",
        loc=0,
        value=clustering.fit_predict(centroid_cells.values),
    )

    # Compute centroids from the previous clutering
    centroids = centroid_cells.groupby("cluster").mean()
    recipients_cells_files = list(
        el for el in Path(cfg.paths.dataset.cells).rglob("*") if el.is_file()
    )

    bags = []
    # Apply centroids to all cells
    for patient in patients:
        patient_cells = [
            recipient
            for recipient in recipients_cells_files
            if recipient.name.startswith(patient)
        ]
        recipient_cells = pd.concat(
            (pd.read_csv(recipient) for recipient in sorted(patient_cells)),
            ignore_index=True,
        )

        recipient_cells.insert(
            loc=0,
            column="cluster",
            value=clustering_utils.apply_centroids(
                recipient_cells[cfg.dgi.training_params.features_list], centroids
            ),
        )
        bag_of_cells = (
            recipient_cells.value_counts("cluster").sort_index()
            / recipient_cells.shape[0]
        )
        bag_of_cells = bag_of_cells.reindex(
            list(range(centroids.shape[0])), fill_value=0
        )

        bags += [
            {
                "bag_of_embedding": bag_of_cells.values,
                "patient": patient,
            }
        ]
    pd.DataFrame(bags).to_parquet(
        global_representations_folder / "bags_of_raw_features",
        partition_cols=["patient"],
    )


def generate_bags_of_embeddings_by_patients(
    cfg: DictConfig,
    global_representations_folder: Path,
    clustering: KMeans,
    training_patients: Iterable[str],
) -> None:
    """
    Generate bags of dgi embeddings for all patients.
    Cells of all ROIs of the same patients are pooled and clustered.
    Centroids of those clusters are used to apply the clustering to all other patients cells dgi embeddings

    All patients are then represented by the frequency of each clusters in the patient cells

    Args:
        cfg (DictConfig)
        global_representations_folder (Path): Folder to save the patient representations
        clustering (KMeans): Clustering to use to identify the bags
        training_patients (Iterable[str]): list of patients to use compute the bags centroids
    """

    # Apply clustering to centroids_rglob cells
    embedding_files = [
        el for el in list(Path(cfg.paths.dataset.embeddings).rglob("*")) if el.is_file()
    ]
    patients = set(file.name.split("_ROI")[0] for file in embedding_files)

    centroid_cells = pd.DataFrame(
        np.concatenate(
            [
                pickle.load(open(embedding_file, "rb"))
                for embedding_file in embedding_files
                for patient in training_patients
                if embedding_file.name.startswith(patient)
            ]
        )
    )
    centroid_cells.insert(
        column="cluster",
        loc=0,
        value=clustering.fit_predict(centroid_cells.values),
    )

    # Compute centroids from the previous clutering
    centroids = centroid_cells.groupby("cluster").mean()

    bags = []
    # Apply centroids to all cells
    for patient in patients:
        patient_files = [
            file for file in embedding_files if file.name.startswith(patient)
        ]
        recipient_cells = pd.DataFrame(
            np.concatenate([pickle.load(open(file, "rb")) for file in patient_files])
        )

        recipient_cells.insert(
            loc=0,
            column="cluster",
            value=clustering_utils.apply_centroids(recipient_cells, centroids),
        )
        bag_of_cells = (
            recipient_cells.value_counts("cluster").sort_index()
            / recipient_cells.shape[0]
        )
        bag_of_cells = bag_of_cells.reindex(
            list(range(centroids.shape[0])), fill_value=0
        )
        bags += [
            {
                "bag_of_embedding": bag_of_cells.values,
                "patient": patient,
            }
        ]
    pd.DataFrame(bags).to_parquet(
        global_representations_folder / f"bags_of_embeddings_{cfg.exp_name}",
        partition_cols=["patient"],
    )


def generate_avg_utag_by_patients(
    cfg: DictConfig, global_representations_folder: Path
) -> None:
    """Load message passing embeddings for all ROIs of a single patient
    save the mean in the representations_folder, (1, d) representation for each patient
    """

    embeddings_files = [
        file
        for file in Path(cfg.paths.dataset.message_passing).rglob("*")
        if file.is_file()
    ]

    patients = set(file.name.split("_ROI")[0] for file in embeddings_files)

    bags_of_embeddings_path = global_representations_folder / "avg_normalized_utag"
    bags = []
    for patient in tqdm(patients):
        patient_files = [
            file for file in embeddings_files if file.name.startswith(patient)
        ]
        embedding = np.concatenate(
            [pickle.load(open(filename, "rb")) for filename in patient_files]
        ).mean(0)

        bags += [
            {
                "bag_of_embedding": embedding,
                "patient": patient,
            }
        ]

    pd.DataFrame(bags).to_parquet(
        bags_of_embeddings_path,
        partition_cols=["patient"],
    )


def generate_bags_of_utag_by_patients(
    cfg: DictConfig,
    global_representations_folder: Path,
    clustering: KMeans,
    training_patients: Iterable[str],
) -> None:
    """
    Generate bags of message passing embeddings for all patients.
    Cells of all ROIs of the same patients are pooled and clustered.
    Centroids of those clusters are used to apply the clustering to all other patients cells message passing embeddings

    All patients are then represented by the frequency of each clusters in the patient cells

    Args:
        cfg (DictConfig)
        global_representations_folder (Path): Folder to save the patient representations
        clustering (KMeans): Clustering to use to identify the bags
        training_patients (Iterable[str]): list of patients to use compute the bags centroids
    """

    # Apply clustering to centroids_rglob cells
    utag_files = [
        el
        for el in list((Path(cfg.paths.dataset.message_passing).rglob("*")))
        if el.is_file()
    ]
    patients = set(file.name.split("_ROI")[0] for file in utag_files)

    centroid_cells = pd.DataFrame(
        np.concatenate(
            [
                pickle.load(open(utag_file, "rb"))
                for utag_file in utag_files
                for patient in training_patients
                if utag_file.name.startswith(patient)
            ]
        )
    )
    centroid_cells.insert(
        column="cluster",
        loc=0,
        value=clustering.fit_predict(centroid_cells.values),
    )

    # Compute centroids from the previous clutering
    centroids = centroid_cells.groupby("cluster").mean()

    bags = []
    # Apply centroids to all filtered_cells
    for patient in patients:
        patient_files = [file for file in utag_files if file.name.startswith(patient)]
        recipient_cells = pd.DataFrame(
            np.concatenate([pickle.load(open(file, "rb")) for file in patient_files])
        )

        recipient_cells.insert(
            loc=0,
            column="cluster",
            value=clustering_utils.apply_centroids(recipient_cells, centroids),
        )
        bag_of_cells = (
            recipient_cells.value_counts("cluster").sort_index()
            / recipient_cells.shape[0]
        )
        bag_of_cells = bag_of_cells.reindex(
            list(range(centroids.shape[0])), fill_value=0
        )

        bags += [
            {
                "bag_of_embedding": bag_of_cells.values,
                "patient": patient,
            }
        ]
    pd.DataFrame(bags).to_parquet(
        global_representations_folder / "bags_of_normalized_utag",
        partition_cols=["patient"],
    )


def generate_representations_by_patients(
    config: DictConfig,
    kmeans_k: int,
    training_patients: Iterable[str],
    output_folder: Path,
) -> None:
    """generate all patient_level representation"""
    clustering_method = KMeans(kmeans_k)
    generate_bags_of_raw_features_by_patients(
        config,
        output_folder,
        clustering_method,
        training_patients=training_patients,
    )

    generate_bags_of_embeddings_by_patients(
        config,
        output_folder,
        clustering_method,
        training_patients=training_patients,
    )

    generate_avg_embeddings_by_patients(config, output_folder)

    generate_bags_of_utag_by_patients(
        config,
        output_folder,
        clustering_method,
        training_patients=training_patients,
    )

    generate_avg_utag_by_patients(config, output_folder)


def generate_bags_of_embeddings(
    cfg: DictConfig,
    global_representations_folder: Path,
    clustering: KMeans,
    training_patients: Iterable[str],
) -> None:
    """
    Generate bags of dgi embeddings for all embeddings in cfg.paths.dataset.embeddings
    embeddings for patients in training_patients are pooled and clustered.
    Centroids of those clusters are used to apply the clustering to all other patients

    All ROIs are then represented by the frequency of each clusters in the ROIs embeddings

    Args:
        cfg (DictConfig)
        global_representations_folder (Path): Folder to save the patient representations
        clustering (KMeans): Clustering to use to identify the bags
        training_patients (Iterable[str]): list of patients to use compute the bags centroids
    """
    # Apply clustering to centroids_rglob cells
    embedding_files = [
        el
        for patient in training_patients
        for el in list(Path(cfg.paths.dataset.embeddings).rglob(f"*/*/{patient}_*"))
    ]

    centroid_cells = pd.DataFrame(
        np.concatenate(
            [
                pickle.load(open(embedding_file, "rb"))
                for embedding_file in embedding_files
            ]
        )
    )
    centroid_cells.insert(
        column="cluster",
        loc=0,
        value=clustering.fit_predict(centroid_cells.values),
    )

    # Compute centroids from the previous clutering
    centroids = centroid_cells.groupby("cluster").mean()
    recipients_embedding_files = list(
        el for el in Path(cfg.paths.dataset.embeddings).rglob("*") if el.is_file()
    )

    bags = []
    # Apply centroids to all cells
    for recipient in recipients_embedding_files:
        recipient_cells = pd.DataFrame(pickle.load(open(recipient, "rb")))

        recipient_cells.insert(
            loc=0,
            column="cluster",
            value=clustering_utils.apply_centroids(recipient_cells, centroids),
        )
        bag_of_cells = (
            recipient_cells.value_counts("cluster").sort_index()
            / recipient_cells.shape[0]
        )
        bag_of_cells = bag_of_cells.reindex(
            list(range(centroids.shape[0])), fill_value=0
        )

        patient = "_".join(recipient.name.split("_")[:-1])
        bags += [
            {
                "bag_of_embedding": bag_of_cells.values,
                "patient": patient,
            }
        ]
    pd.DataFrame(bags).to_parquet(
        global_representations_folder / f"bags_of_embeddings_{cfg.exp_name}",
        partition_cols=["patient"],
    )


def generate_avg_utag(cfg: DictConfig, global_representations_folder: Path) -> None:
    """load all message passing embeddings from cfg.paths.dataset.message_passing
    save the mean in the representations_folder"""

    utag_files = [
        file
        for file in Path(cfg.paths.dataset.message_passing).rglob("*")
        if file.is_file()
    ]

    bags_of_utag_path = global_representations_folder / "avg_normalized_utag"
    bags = []
    for filename in tqdm(utag_files):
        with open(filename, "rb") as file:
            embedding = pickle.load(file).mean(0)

        patient = "_".join(filename.name.split("_message_passing")[:-1])
        bags += [
            {
                "bag_of_embedding": embedding,
                "patient": patient,
            }
        ]

    pd.DataFrame(bags).to_parquet(
        bags_of_utag_path,
        partition_cols=["patient"],
    )


def generate_bags_of_utag(
    cfg: DictConfig,
    global_representations_folder: Path,
    clustering: KMeans,
    training_patients: Iterable[str],
) -> None:
    """
    Generate bags of message passing embeddings for all embeddings in cfg.paths.dataset.embeddings
    embeddings for patients in training_patients are pooled and clustered.
    Centroids of those clusters are used to apply the clustering to all other patients

    All ROIs are then represented by the frequency of each clusters in the ROIs embeddings

    Args:
        cfg (DictConfig)
        global_representations_folder (Path): Folder to save the patient representations
        clustering (KMeans): Clustering to use to identify the bags
        training_patients (Iterable[str]): list of patients to use compute the bags centroids
    """
    # Apply clustering to centroids_rglob cells
    embedding_files = [
        el
        for patient in training_patients
        for el in list((Path(cfg.paths.dataset.message_passing)).rglob("*"))
        if el.stem.startswith(f"{patient}_")
    ]

    centroid_cells = pd.DataFrame(
        np.concatenate(
            [
                pickle.load(open(embedding_file, "rb"))
                for embedding_file in embedding_files
            ]
        )
    )
    centroid_cells.insert(
        column="cluster",
        loc=0,
        value=clustering.fit_predict(centroid_cells.values),
    )

    # Compute centroids from the previous clutering
    centroids = centroid_cells.groupby("cluster").mean()
    recipients_embedding_files = list(
        el for el in Path(cfg.paths.dataset.message_passing).rglob("*") if el.is_file()
    )

    bags = []
    # Apply centroids to all cells
    for recipient in recipients_embedding_files:
        recipient_cells = pd.DataFrame(pickle.load(open(recipient, "rb")))

        recipient_cells.insert(
            loc=0,
            column="cluster",
            value=clustering_utils.apply_centroids(recipient_cells, centroids),
        )
        bag_of_cells = (
            recipient_cells.value_counts("cluster").sort_index()
            / recipient_cells.shape[0]
        )
        bag_of_cells = bag_of_cells.reindex(
            list(range(centroids.shape[0])), fill_value=0
        )

        patient = "_".join(recipient.name.split("_message_passing")[:-1])
        bags += [
            {
                "bag_of_embedding": bag_of_cells.values,
                "patient": patient,
            }
        ]
    pd.DataFrame(bags).to_parquet(
        global_representations_folder / "bags_of_normalized_utag",
        partition_cols=["patient"],
    )


def generate_representations(
    config: DictConfig,
    kmeans_k: int,
    training_patients: Iterable[str],
    output_folder: Path,
) -> None:
    """generate all roi_level representations"""
    clustering_method = KMeans(kmeans_k)
    generate_bags_of_raw_features(
        config,
        output_folder,
        clustering_method,
        training_patients=training_patients,
    )
    generate_bags_of_utag(
        config,
        output_folder,
        clustering_method,
        training_patients=training_patients,
    )

    generate_avg_utag(config, output_folder)

    generate_bags_of_embeddings(
        config,
        output_folder,
        clustering_method,
        training_patients=training_patients,
    )

    generate_avg_embeddings(config, output_folder)


def load_representations(
    path: Path, training_patients: Iterable[str], test_patients: Iterable[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """load the parquet file in path and split btwn train and test patients"""
    data = pd.read_parquet(path)
    return (
        data[data["patient"].isin(training_patients)],
        data[data["patient"].isin(test_patients)],
    )


def linear_classification(
    output_folder: Path,
    training_patients: List[str],
    test_patients: List[str],
    gts_df: pd.DataFrame,
    label_col: str,
    exp_name: str,
) -> pd.DataFrame:
    """Graph level logistic regression using the graph-level representations

    Args:
        output_folder (Path): Path to store the results
        training_patients (List[str]): List of training patients (for both logistic regression and bags computations)
        test_patients (List[str]): List of patients to compute metrics
        gts_df (pd.DataFrame): Dataframe with graph level annotations
        label_col (str): column to use for label in gts_df
        exp_name (str): id for the experiment (which representation)

    Returns:
        pd.DataFrame: df with test metrics results (with predictions probas and gts)
    """
    results = []
    classes = list(gts_df[label_col].unique())
    for representation in [
        "bags_of_normalized_utag",
        f"bags_of_embeddings_{exp_name}",
        "bags_of_raw_features",
        f"avg_embeddings_{exp_name}",
        "avg_normalized_utag",
    ]:
        training_set, test_set = load_representations(
            output_folder / representation,
            training_patients,
            test_patients,
        )
        training_set = pd.merge(training_set, gts_df, on="patient")
        test_set = pd.merge(test_set, gts_df, on="patient")
        results += [
            pd.DataFrame(
                multiclass_utils.multiclass_classification(
                    training_set,
                    test_set,
                    representation,
                    classes=classes,
                    label_col=label_col,
                )
            )
        ]
    return pd.concat(results, ignore_index=True)


def classification_pipeline(
    cfg: DictConfig,
    gts_df: pd.DataFrame,
    training_patients: List[str],
    test_patients: List[str],
    output_folder: Path,
    label_col: str,
) -> pd.DataFrame:
    """Generate all ROI-level representations and run the log reg pipeline

    Args:
        cfg (DictConfig)
        gts_df (pd.DataFrame): dataframe with gt label for each graph
        training_patients (List[str]): list of patients to use for training
        test_patients (List[str]): test patients ids
        output_folder (Path): Folder to store test metrics
        label_col (str): column of gts_df to get label
    """
    results = []

    generate_representations(
        cfg,
        kmeans_k=cfg.scripts.evaluations.graph_level_experiments.k_bags,
        training_patients=training_patients,
        output_folder=output_folder,
    )

    results += [
        linear_classification(
            output_folder,
            training_patients,
            test_patients,
            gts_df,
            label_col=label_col,
            exp_name=cfg.exp_name,
        )
    ]

    return pd.concat(results, ignore_index=True)


def patient_level_classification_pipeline(
    cfg: DictConfig,
    gts_df: pd.DataFrame,
    training_patients: List[str],
    test_patients: List[str],
    output_folder: Path,
    label_col: str,
) -> pd.DataFrame:
    """Generate all patient-level representations and run the log reg pipeline

    Args:
        cfg (DictConfig)
        gts_df (pd.DataFrame): dataframe with gt label for each graph
        training_patients (List[str]): list of patients to use for training
        test_patients (List[str]): test patients ids
        output_folder (Path): Folder to store test metrics
        label_col (str): column of gts_df to get label
    """
    results = []

    generate_representations_by_patients(
        cfg,
        kmeans_k=cfg.scripts.evaluations.graph_level_experiments.k_bags,
        training_patients=training_patients,
        output_folder=output_folder,
    )

    results += [
        linear_classification(
            output_folder,
            training_patients,
            test_patients,
            gts_df,
            label_col=label_col,
            exp_name=cfg.exp_name,
        )
    ]

    return pd.concat(results, ignore_index=True)
