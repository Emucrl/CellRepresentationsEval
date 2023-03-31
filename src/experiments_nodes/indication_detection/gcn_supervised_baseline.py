from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch_geometric.data import Data

from src.utils.graph_dataset import GraphDataset
from src.utils.graph_utils import adj_feat_to_data, load_graph


class IndicationGraphDataset(GraphDataset):
    """Indication graph dataset"""

    def __init__(
        self,
        data_path: Path,
        features_list: List,
        patients: List,
        classes: List[str],
        gts_df: pd.DataFrame,
    ) -> None:
        """Graph dataset for supervised gcn training
        Args:
            data_path (Path): Path to folder with graphs
            features_list (List): list of features to consider
            patients (List): Patients to use (files are stored as PATIENTID_adjacency.p/PATIENTID_features.p)
            classes (List[str]): List of classes considered
            gts_df (pd.DataFrame): Dataframe with patient and indication columns for groundtruth
        """
        super().__init__(data_path, features_list)
        self.gts_df = gts_df
        self.graphs_paths = [
            graph
            for graph in self.graphs_paths  # type: ignore
            for patient in patients
            if f"{patient}_" in graph.as_posix()
            and any(
                f"indication={indication}" in graph.as_posix() for indication in classes
            )
        ]
        self.classes = classes

    def _lazy_get_item(self, idx: int) -> Data:
        """Generate a pyg.Data object with the graph and the corresponding label"""
        path = self.graphs_paths[idx]
        patient = path.stem.split("_{")[0]
        adjacency, features = load_graph(str(path))
        data = adj_feat_to_data(adjacency, features, self.features_list)
        data.y = torch.tensor(
            self.classes.index(
                self.gts_df.query(f"patient=='{patient}'")["indication"].values[0]
            ),
            dtype=torch.long,
        )
        return data
