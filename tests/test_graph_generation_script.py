"""test for graph generation script"""
import os
from pathlib import Path

from src.data_processing_nodes.patient_graph_generation import generate_all_graphs


def test_generate_graphs(tmp_path):
    """test the generation of graphs for all patients"""
    generate_all_graphs(
        max_edge_size=10,
        source_data_path=Path("tests/test_data/ready_for_graph_creation"),
        target_path=tmp_path,
        n_splits=1,
    )
    generated_files = list(tmp_path.iterdir())
    assert len(generated_files) == 2 * len(
        os.listdir("tests/test_data/ready_for_graph_creation")
    )
    assert len([file for file in generated_files if "adj" in str(file)]) == len(
        [file for file in generated_files if "feat" in str(file)]
    )
