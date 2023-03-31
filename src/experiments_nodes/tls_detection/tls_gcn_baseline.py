# %%
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.utils import cell_level_classification_utils, graph_dataset, graph_utils


class GraphNodeLabel(graph_dataset.GraphDataset):
    """Node level dataset for tls predictions"""

    def __init__(
        self,
        data_path: Path,
        patients: List[str],
        cfg: DictConfig,
    ) -> None:
        """Selects graphs filenames for the required patients"""
        super().__init__(data_path, cfg.dgi.training_params.features_list)
        self.cfg = cfg
        lst_graph_path = []
        for patient in patients:
            for graph_path in self.graphs_paths:  # type: ignore
                if patient in str(graph_path):
                    lst_graph_path.append(graph_path)

        self.graphs_paths = lst_graph_path

    def _lazy_get_item(self, idx: int) -> Data:
        """Generates a pyg.Data object
        Cell-level gt is defined with the ROIs tls binary mask
        """
        path = self.graphs_paths[idx]
        filename = str(path)
        adjacency, features = graph_utils.load_graph(filename)
        patient = os.path.basename(filename)
        i = patient.find("_{}")
        patient = patient[:i]
        gts = cell_level_classification_utils.get_tls_gt(patient, self.cfg)
        gts = torch.tensor(np.array(gts))
        data = graph_utils.adj_feat_to_data(adjacency, features, self.features_list)
        data.y = gts

        return data
