"""Utils related to binary mask creation, usage"""

from typing import Tuple

import numpy as np
import pandas as pd
from skimage import morphology


def apply_mask_cells_df(cells_df: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    """apply mask to cells_df
    cells_df must have x, y columns"""

    return cells_df.apply(lambda row: int(0 < mask[int(row["x"]), int(row["y"])]), 1)


def get_tls_mask(
    roi_cells: pd.DataFrame,
    tls_cluster_index: int,
    mask_size: Tuple[int, int] = (1000, 1000),
) -> np.ndarray:
    """Generate binary mask from cell df and cluster index using morphology operations
    Args:
        roi_cells (pd.DataFrame): cells df. should have a cluster col and x,y
        tls_cluster_index (int): id of the cluster to contour
        mask_size (Tuple[int, int]): size of the image ~offset+roi_size
    Returns:
        np.ndarray: binary mask aroung the cluster with id tls_cluster_id
    """
    zeros = np.zeros(mask_size)
    cells = roi_cells.query(f"cluster=={tls_cluster_index}")[["x", "y"]].values.astype(
        int
    )
    zeros[cells[:, 0], cells[:, 1]] = 1

    dilated = morphology.binary_dilation(zeros, np.ones((10, 10)))

    closed = morphology.binary_closing(dilated, morphology.disk(3))

    opened = morphology.binary_dilation(closed, morphology.disk(2))

    no_holes = morphology.remove_small_holes(opened, 200)
    return no_holes
