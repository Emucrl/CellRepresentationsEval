"""
scripts for subgraph sampling
"""
import os
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

from src.utils.graph_utils import sample_subgraphs_from_cells_df


def generate_subgraphs_all_patients(
    patients_graphs_folder: Path,
    subgraphs_config: Dict[str, Any],
    save_path: Path,
) -> None:
    """run num_subgraphs_per_graph times the random graph sampling for each csv in patients
    tries to append the corresponding samples indices to existing save_path/samples_indices.pkl
    create the file otherwise

    All generated graphs are stored in save_path as well as samples_indices.pkl (pd.DataFrame)
    samples_indices.pkl allows to sample non-overlapping subgraphs, the threshold being defined
    in conf/graph_creations/subgraphs

    subgraphs_config needs: overlap_threshold/d_to_centroid/num_subgraphs_per_graph/max_edge_size

    """
    save_path.mkdir(exist_ok=True, parents=True)
    if not (save_path / "samples_indices.pkl").exists():
        samples_indices_df = pd.DataFrame(
            [], columns=["slide_id", "sample_id", "nodes_indices"]
        )
    else:
        with open(
            save_path / "samples_indices.pkl",
            "rb",
        ) as file:
            samples_indices_df = pickle.load(file)

    patients_files = [
        file for file in patients_graphs_folder.rglob("*") if file.name.endswith(".csv")
    ]

    for patient_file in tqdm(patients_files):
        cells = pd.read_csv(patient_file)

        # find previous samples for this graph
        samples_indices = list(
            samples_indices_df[samples_indices_df["slide_id"] == patient_file.stem]
            .loc[:, "nodes_indices"]
            .values
        )

        for _ in tqdm(range(subgraphs_config["num_subgraphs_per_graph"])):
            new_adj, new_feats, samples_indices = sample_subgraphs_from_cells_df(
                cells,
                samples_indices,
                max_edge_size=subgraphs_config["max_edge_size"],
                d_to_centroid=subgraphs_config["d_to_centroid"],
                overlap_threshold=subgraphs_config["overlap_threshold"],
            )

            if new_adj is not None:
                savefile = os.path.join(
                    save_path,
                    f"{ patient_file.stem}_random{len(samples_indices)-1}_{{}}.p",
                )
                with open(
                    savefile.format("adjacency"),
                    "wb",
                ) as write:
                    pickle.dump(new_adj, write)
                with open(
                    savefile.format("features"),
                    "wb",
                ) as write:
                    pickle.dump(new_feats, write)

                samples_indices_df = pd.concat(
                    [
                        samples_indices_df,
                        pd.DataFrame(
                            {
                                "slide_id": [patient_file.stem],
                                "sample_id": [len(samples_indices) - 1],
                                "nodes_indices": [samples_indices[-1]],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

        with open(
            save_path / "samples_indices.pkl",
            "wb",
        ) as write:
            pickle.dump(samples_indices_df, write)
