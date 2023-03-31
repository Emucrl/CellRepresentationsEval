"""generate masks"""
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from skimage import io
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from src.utils import masking_utils, utils


def density_based_clustering_tls(
    df_cells: pd.DataFrame, min_samples: int = 3, eps: int = 10
) -> pd.DataFrame:
    """apply dbscan to the df_cells
    A cluster column is added to df_cells. value of -1 means not clustered, otherwise cluster id

    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html for params tuning

    Returns:
        pd.DataFrame: df_cells with an additional "cluster" column
    """
    df_cells.insert(
        loc=df_cells.shape[1],
        column="cluster",
        value=DBSCAN(
            min_samples=min_samples,
            eps=eps,
        ).fit_predict(df_cells[["x", "y"]]),
    )
    return df_cells


def generate_masks(cfg: DictConfig) -> None:
    """generate masks and save to cfg["paths"][dataset]["tls_masks"]

    Positive cells are first identified with a pandas query defined in cfg.scripts.evaluations.bcells_query
    Then a density-based clustering is applied to identify dense aggregates
    Finally, a binary masks is generated for each cluster that meets the filtering criteria

    Args:
        cfg (DictConfig)
    """
    target_folder = Path(
        cfg["paths"][cfg.scripts.evaluations.eval_dataset]["tls_masks"]
    )
    target_folder.mkdir(parents=True, exist_ok=True)

    # Load all source data
    cells_files = sorted(
        [
            file
            for file in Path(
                cfg["paths"][cfg.scripts.evaluations.eval_dataset]["cells"]
            ).rglob("*")
            if file.is_file()
        ]
    )
    for cells_file in tqdm(cells_files):
        patient_unfiltered_cells = pd.read_csv(cells_file)

        # Apply density based clustering to find TLS (~dense aggregate of bcells)
        b_cells = patient_unfiltered_cells.query(cfg.scripts.evaluations.bcells_query)
        b_cells = density_based_clustering_tls(b_cells)

        for cluster_index in b_cells["cluster"].unique():
            # Remove cluters too small to be TLS
            if (
                b_cells.query(f"cluster=={cluster_index}").shape[0]
                < cfg.scripts.evaluations.tls_min_size
            ):
                b_cells.loc[
                    b_cells[b_cells["cluster"] == cluster_index].index,
                    "cluster",
                ] = -1

        if len(b_cells["cluster"].unique()) > 1:
            # Apply morpho operations to generate mask from cluster cells coordinates
            # Sum mask of all tls to have one mask for each patient
            tls_mask = np.sum(
                [
                    masking_utils.get_tls_mask(b_cells, tls_id)
                    for tls_id in b_cells["cluster"].unique()
                    if tls_id != -1
                ],
                axis=0,
            )
            filename = utils.swap_root_directory(
                Path(cfg["paths"][cfg.scripts.evaluations.eval_dataset]["cells"]),
                target_folder,
                cells_file,
            )
            filename.parent.mkdir(exist_ok=True, parents=True)
            io.imsave(
                filename.with_name(f"{filename.stem.replace('_cells', '_mask')}.jpg"),
                255 * tls_mask.astype(np.uint8),
            )
