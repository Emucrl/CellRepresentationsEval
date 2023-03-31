"""GraphDataset for cell level tumor classification"""
from pathlib import Path
from typing import List

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.utils import graph_dataset, graph_utils, utils


class GraphNodeLabel(graph_dataset.GraphDataset):
    """Node level dataset for tumor predictions"""

    def __init__(self, data_path: Path, patients: List[str], cfg: DictConfig) -> None:
        """Selects graphs filenames for the required patients"""
        super().__init__(data_path, cfg.dgi.training_params.features_list)
        self.graphs_paths = [
            graph
            for patient in patients
            for graph in self.graphs_paths  # type: ignore
            if f"{patient}_" in graph.as_posix()
        ]

    def _lazy_get_item(self, idx: int) -> Data:
        """Generates a pyg.Data object
        Cell-level gt is defined with the tumour_mask column in the features matrix
        """
        path = self.graphs_paths[idx]
        filename = str(path)
        adjacency, features = graph_utils.load_graph(filename)

        data = graph_utils.adj_feat_to_data(adjacency, features, self.features_list)
        data.y = torch.tensor(utils.get_is_tumor(features)["is_tumour"].values).long()
        return data
