"""Script functions revolving around generating graphs
"""
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils import utils
from src.utils.graph_utils import generate_graph_from_splits


def generate_all_graphs(
    max_edge_size: int, source_data_path: Path, target_path: Path, n_splits: int
) -> None:
    """generate graphs, using generate_graph_from_splits, for all csv
    located in source_data_path and stores them in target_path

    n_split params enables to reduce the computational workload by splitting the csv,
    generating a graph for each split and merging. Useful for cells derived from WSIs

    Args:
        max_edge_size (int):max distance between nodes to have an edge.
        source_data_path (Path): location of source csv's.
        target_path (Path): location to save generated graphs.
        n_split (int): #splits of the 2D space to reduce computational workload
    """
    csv_filenames = [
        filename
        for filename in source_data_path.rglob("*")
        if filename.name.endswith(".csv")
    ]

    csv_filenames = sorted(csv_filenames, key=lambda x: x.stat().st_size)

    for _, csv_filename in tqdm(enumerate(csv_filenames)):
        patient = csv_filename.stem.split("_cells")[0]

        cells = pd.read_csv(csv_filename)

        adjacency_matrix, features = generate_graph_from_splits(
            cells, max_edge_size=max_edge_size, n_splits=n_splits
        )

        target_folder = utils.swap_root_directory(
            source_data_path, target_path, csv_filename
        ).parent
        target_folder.mkdir(exist_ok=True, parents=True)

        with open(
            target_folder / f"{patient}_adjacency.p",
            "wb",
        ) as adjacency_file:
            pickle.dump(adjacency_matrix, adjacency_file)

        with open(
            target_folder / f"{patient}_features.p",
            "wb",
        ) as features_file:
            pickle.dump(features, features_file)
