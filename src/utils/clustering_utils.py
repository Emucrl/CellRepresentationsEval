"""clustering utils"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


def find_centroids(embeddings: np.ndarray, clusters: np.ndarray) -> pd.DataFrame:
    """Generate clustering centroids for each cluster (avg embedding for each cluster)

    Args:
        embeddings (np.ndarray): cell representations that were used for clustering (nxd)
        clusters (np.ndarray): cluster id for each cell (n,)

    Returns:
        pd.DataFrame: (k, d) df with centroid representations for each clusters
    """
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.insert(loc=0, value=clusters, column="cluster")
    centroids = embeddings_df.groupby("cluster").mean()
    centroids.columns = [str(column) for column in centroids.columns]
    return centroids


def apply_centroids(embeddings: np.ndarray, centroids: pd.DataFrame) -> np.ndarray:
    """For each cell (row) in embeddings, identify the closest centroid
    which will be the cluster assigned to the cell

    Args:
        embeddings (np.ndarray): cells to cluster representations (n,d)
        centroids (pd.DataFrame): centroids representations (k,d)

    Returns:
        np.ndarray: closest centroid id for each cell in embeddings (n,1)
    """
    return euclidean_distances(embeddings, centroids).argmin(1)
