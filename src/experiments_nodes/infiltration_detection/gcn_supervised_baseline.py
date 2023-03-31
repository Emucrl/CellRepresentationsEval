from pathlib import Path
from typing import List

import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from src.utils.graph_dataset import GraphDataset
from src.utils.graph_utils import adj_feat_to_data, load_graph


class InfiltrationGraphDataset(GraphDataset):
    """Infiltration graph dataset"""

    def __init__(
        self,
        data_path: Path,
        features_list: List,
        patients: List,
        gts_df: pd.DataFrame,
        classes: List[str],
    ) -> None:
        """Infiltration scores are given at the patient level in our data
        while each patient may have multiple IMC ROIs

        This datasets groups all graphs for the same patients (PATIENTID_ROIX_adjacency.p)
        to create a single graph by patient

        If '_ROI' is not in the filenames, graphs will be treated indepently

        Args:
            data_path (Path): Folder with graphs
            features_list (List): features to consider
            patients (List): patients to consider
            gts_df (pd.DataFrame): groundtruth. Should have a patient column and an infiltration columns
            classes (List[str]): list of possible classes
        """
        super().__init__(data_path, features_list)
        self.gts_df = gts_df
        self.patients = patients
        self.graphs_paths_by_patients = [
            sorted([el for el in self.graphs_paths if el.stem.startswith(patient)])
            for patient in patients
        ]
        assert len(self.graphs_paths_by_patients) == len(self.patients)
        self.gts_df = gts_df
        self.classes = classes

    def _lazy_get_item(self, idx: int) -> Data:
        """Concatenate all graphs that have the same patient id (PATIENTID_ROIX_adjacency.p)"""
        paths = self.graphs_paths_by_patients[idx]
        patient = paths[0].stem.split("_ROI")[0]
        roi_graphs = [load_graph(str(path)) for path in paths]
        patient_adj = sp.block_diag((graph[0] for graph in roi_graphs), "csr")
        patient_feats = pd.concat((graph[1] for graph in roi_graphs), ignore_index=True)
        data = adj_feat_to_data(patient_adj, patient_feats, self.features_list)
        data.y = torch.tensor(
            self.classes.index(
                self.gts_df.query(f"patient=='{patient}'")["infiltration"].values[0]
            ),
            dtype=torch.long,
        )
        return data

    def __len__(self) -> int:
        """Returns length of the dataset."""

        return len(self.patients)
