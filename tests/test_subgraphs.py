"""tests for subgraphs script"""
import os
import pickle
from pathlib import Path

import pytest

from src.data_processing_nodes.subgraphs import generate_subgraphs_all_patients


@pytest.fixture(name="cfg")
def generate_config():
    """config with overlap threshold =1"""
    return {
        "files": Path("tests/test_data/ready_for_graph_creation"),
        "subgraphs": {
            "d_to_centroid": 500,
            "num_subgraphs_per_graph": 10,
            "overlap_threshold": 1,
            "max_edge_size": 300,
        },
    }


@pytest.fixture(name="cfg_overlap_check")
def generate_config_overlap():
    """config with overlap threshold <1"""
    return {
        "files": Path("tests/test_data/ready_for_graph_creation"),
        "subgraphs": {
            "d_to_centroid": 500,
            "num_subgraphs_per_graph": 10,
            "overlap_threshold": 0.5,
            "max_edge_size": 300,
        },
    }


def test_generate_subgraphs_all_test(cfg, tmp_path):
    """test that the right amount of graphs are generated without checking overlap"""
    generate_subgraphs_all_patients(
        patients_graphs_folder=cfg["files"],
        subgraphs_config=cfg["subgraphs"],
        save_path=tmp_path
        / str(cfg["subgraphs"]["max_edge_size"])
        / f'subgraphsize_{cfg["subgraphs"]["d_to_centroid"]}',
    )
    generated = os.listdir(
        tmp_path
        / str(cfg["subgraphs"]["max_edge_size"])
        / f'subgraphsize_{cfg["subgraphs"]["d_to_centroid"]}'
    )
    print(cfg["subgraphs"]["num_subgraphs_per_graph"])
    assert (
        len(generated)
        == 2
        * len(list(cfg["files"].rglob("*")))
        * cfg["subgraphs"]["num_subgraphs_per_graph"]
        + 1
    )
    assert "samples_indices.pkl" in generated
    with open(
        tmp_path
        / str(cfg["subgraphs"]["max_edge_size"])
        / f'subgraphsize_{cfg["subgraphs"]["d_to_centroid"]}'
        / "samples_indices.pkl",
        "rb",
    ) as file:
        samples = pickle.load(file)
    assert (
        samples.shape[0]
        == len(list(cfg["files"].rglob("*")))
        * cfg["subgraphs"]["num_subgraphs_per_graph"]
    )


def test_generate_subgraphs_ovelap_all_test(cfg_overlap_check, tmp_path):
    """test that overlap reduces the number of generated graphs and that samples_idx.pkl
    has the right number of indexes"""
    generate_subgraphs_all_patients(
        patients_graphs_folder=cfg_overlap_check["files"],
        subgraphs_config=cfg_overlap_check["subgraphs"],
        save_path=tmp_path
        / str(cfg_overlap_check["subgraphs"]["max_edge_size"])
        / f'subgraphsize_{cfg_overlap_check["subgraphs"]["d_to_centroid"]}',
    )
    generated = os.listdir(
        tmp_path
        / str(cfg_overlap_check["subgraphs"]["max_edge_size"])
        / f'subgraphsize_{cfg_overlap_check["subgraphs"]["d_to_centroid"]}'
    )
    assert (
        len(generated)
        < 2 * cfg_overlap_check["subgraphs"]["num_subgraphs_per_graph"] + 1
    )
    assert "samples_indices.pkl" in generated

    with open(
        tmp_path
        / str(cfg_overlap_check["subgraphs"]["max_edge_size"])
        / f'subgraphsize_{cfg_overlap_check["subgraphs"]["d_to_centroid"]}'
        / "samples_indices.pkl",
        "rb",
    ) as file:
        samples = pickle.load(file)
    assert samples.shape[0] == (len(generated) - 1) / 2
