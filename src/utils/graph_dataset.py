""" Graph dataset """

from pathlib import Path
from typing import List

from torch.utils.data import Dataset
from torch_geometric.data.data import BaseData

from src.utils.graph_utils import (
    adj_feat_to_data,
    get_graphs_names_in_folder,
    load_graph,
)


class GraphDataset(Dataset):
    """Graph dataset."""

    def __init__(
        self,
        data_path: Path,
        features_list: List,
        limit_size: int = 10000,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.limit_size = limit_size
        self.graphs_paths = self.__load_graphs_paths__()

        self.features_list = features_list

    def __load_graphs_paths__(self) -> List[Path]:
        """Look for all the graph files in self.data_path
        Only keep those with edge_{max_edge_size} in the filename
        Only keep those that have both adjacency and feature matrix
        Sort by adjacency filesize
        """

        return get_graphs_names_in_folder(self.data_path, limit_size=self.limit_size)

    def __len__(self) -> int:
        """Returns length of the dataset."""
        return len(self.graphs_paths)

    def _lazy_get_item(self, idx: int) -> BaseData:
        path = self.graphs_paths[idx]
        filename = str(path)
        adjacency, features = load_graph(filename)
        return adj_feat_to_data(adjacency, features, self.features_list)

    def __getitem__(self, idx: int) -> BaseData:
        return self._lazy_get_item(idx)

    def update_features_list(self, new_features_list: List[str]) -> None:
        """enables to update the feature list of the graphDataset after creation
        useful for experimenting without having to recreate a new dataset object

        The new feature list should be a subset of the current one.
        For experiments purposes, create the graphDataset with all possible features
        and update for each experiments

        Args:
            new_features_list (List[str]): list of features to keep
        """
        self.features_list = new_features_list
