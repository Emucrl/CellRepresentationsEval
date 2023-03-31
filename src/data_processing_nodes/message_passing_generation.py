"""message passing generation node"""
import pickle
from pathlib import Path
from typing import List

import numpy as np

from src.utils import graph_dataset, graph_utils, utils


def message_passing_generation_node(
    graphs_path: Path, target_path: Path, features_list: List[str]
) -> None:
    """Read all graphs in graphs_path and generate the message passing in target_path

    Args:
        graphs_path (Path)
        target_path (Path)
        features_list (List[str])
    """
    dataset = graph_dataset.GraphDataset(
        data_path=graphs_path,
        features_list=features_list,
    )

    for _, graph_filename in enumerate(dataset.graphs_paths):
        adj, feats = graph_utils.load_graph(str(graph_filename))

        message_passing = np.matmul(
            adj.todense(),
            feats.loc[:, features_list].values,
        )
        degree_normalized_message_passing = np.array(
            message_passing
            / (np.array((adj.todense() > 0).sum(0))[0] + 1)[:, np.newaxis]
        )
        target_filename = Path(
            utils.swap_root_directory(graphs_path, target_path, graph_filename)
            .as_posix()
            .format("message_passing")
        )

        target_filename.parent.mkdir(exist_ok=True, parents=True)
        with open(
            target_filename,
            "wb",
        ) as file:
            pickle.dump(
                degree_normalized_message_passing,
                file,
            )
