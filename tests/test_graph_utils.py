"""graph utils tests"""
import random

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
import torch_geometric

from src.utils.graph_utils import (
    add_degree_to_features,
    adj_feat_to_data,
    generate_adj_from_csv_torch,
    generate_graph_from_csv,
    generate_graph_from_splits,
    sample_subgraphs_from_cells_df,
)


@pytest.fixture(name="simple_cells")
def fixture_simple_cells():
    """cells df"""
    return pd.DataFrame(
        {
            "x": [0, 0, 299, 299, 150],
            "y": [0, 299, 0, 299, 150],
            "feature": ["2", 3, "e", "5", None],
        }
    )


@pytest.fixture(name="simple_cells_float")
def fixture_simple_cells_float():
    """cells df with float features"""
    return pd.DataFrame(
        {
            "x": [0, 0, 299, 299, 150],
            "y": [0, 299, 0, 299, 150],
            "feature": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


@pytest.fixture(name="adjacency_matrix")
def generate_adjacency():
    """adj matrix"""
    return sp.csr_matrix(
        np.array(
            [
                [0.0, 0.00334448, 0.00334448, 0.0, 0.00471405],
                [0.00334448, 0.0, 0.0, 0.00334448, 0.00472978],
                [0.00334448, 0.0, 0.0, 0.00334448, 0.00472978],
                [0.0, 0.00334448, 0.00334448, 0.0, 0.00474568],
                [0.00471405, 0.00472978, 0.00472978, 0.00474568, 0.0],
            ]
        )
    )


def test_add_degree_feature(simple_cells_float, adjacency_matrix):
    """test computing degree and adding the column to pd"""
    feats_with_deg = add_degree_to_features(simple_cells_float, adjacency_matrix)
    assert "degree" in feats_with_deg.columns
    assert list(feats_with_deg.degree.values) == [3, 3, 3, 3, 4]
    assert feats_with_deg.normed_degree.mean() == pytest.approx(0)
    assert feats_with_deg.normed_degree.std() == pytest.approx(1, rel=0.2)


def test_graph_creation(simple_cells):
    """tets generate_graph_from_csv"""
    adj, features = generate_graph_from_csv(simple_cells, 300)
    verif_adj = np.array(
        [
            [0.0, 0.00334448, 0.00334448, 0.0, 0.00471405],
            [0.00334448, 0.0, 0.0, 0.00334448, 0.00472978],
            [0.00334448, 0.0, 0.0, 0.00334448, 0.00472978],
            [0.0, 0.00334448, 0.00334448, 0.0, 0.00474568],
            [0.00471405, 0.00472978, 0.00472978, 0.00474568, 0.0],
        ]
    )
    verif_feats = np.array(
        [
            [0, 0, "2", 3],
            [0, 299, 3, 3],
            [299, 0, "e", 3],
            [299, 299, "5", 3],
            [150, 150, None, 4],
        ]
    )

    assert np.array_equal(adj.toarray().round(decimals=8), verif_adj.round(decimals=8))
    assert np.array_equal(np.array(features), verif_feats)


def test_geometric_data(simple_cells_float):
    """tets generate_graph_from_csv"""
    adj, features = generate_graph_from_csv(simple_cells_float, 300)
    data = adj_feat_to_data(adj, features, ["feature", "deg"])
    data_adj = torch_geometric.utils.to_scipy_sparse_matrix(
        data.edge_index, data.edge_weight
    )

    assert np.array_equal(
        adj.todense().astype(float).round(decimals=7),
        data_adj.todense().astype(float).round(decimals=7),
    )
    assert np.array_equal(
        np.array(features.loc[:, ["feature", "deg"]]), np.array(data.x)
    )


def test_graph_creation_torch(simple_cells_float):
    """tets generate_graph_from_csv"""
    adj = generate_adj_from_csv_torch(simple_cells_float.astype(float), 300)
    verif_adj = np.array(
        [
            [1.0, 0.00334448, 0.00334448, 0.0, 0.00471405],
            [0.00334448, 1.0, 0.0, 0.00334448, 0.00472978],
            [0.00334448, 0.0, 1.0, 0.00334448, 0.00472978],
            [0.0, 0.00334448, 0.00334448, 1.0, 0.00474568],
            [0.00471405, 0.00472978, 0.00472978, 0.00474568, 1.0],
        ]
    )

    np.testing.assert_almost_equal(adj.numpy(), verif_adj, decimal=4)
    cells_df = pd.read_csv(
        "tests/test_data/ready_for_graph_creation/10074349_cells.csv"
    )

    adj = generate_adj_from_csv_torch(cells_df, max_edge_size=300)
    gt = generate_graph_from_csv(cells_df, max_edge_size=300)[0].A + np.identity(
        len(adj)
    )
    np.testing.assert_almost_equal(adj, gt, decimal=4)


def test_graph_creation_splits():
    """tets generate_graph_from_splits"""

    cells_df = pd.read_csv(
        "tests/test_data/ready_for_graph_creation/10074349_cells.csv"
    )
    gt = generate_graph_from_csv(cells_df, max_edge_size=100)[0].A + np.identity(
        len(cells_df)
    )
    adj = generate_graph_from_splits(cells_df, max_edge_size=100, n_splits=1)[0]
    np.testing.assert_almost_equal(adj.A, gt, decimal=4)
    adj = generate_graph_from_splits(cells_df, max_edge_size=100, n_splits=10)[0]
    np.testing.assert_almost_equal(adj.A, gt, decimal=4)
    adj = generate_graph_from_splits(cells_df, max_edge_size=100, n_splits=3)[0]
    np.testing.assert_almost_equal(adj.A, gt, decimal=4)


def test_graph_creation_torch_empty(simple_cells_float):
    """tets generate_graph_from_csv"""
    adj = generate_adj_from_csv_torch(simple_cells_float.astype(float), 100)
    verif_adj = np.identity(5)

    np.testing.assert_almost_equal(adj.numpy(), verif_adj, decimal=4)


@pytest.mark.parametrize(
    ("sample_config", "samples_indices", "expected_indices"),
    (
        (
            {"d_to_centroid": 450, "max_edge_size": 450, "overlap_threshold": 1},
            [],
            [{0, 1, 2, 3, 4}],
        ),
        (
            {"d_to_centroid": 450, "max_edge_size": 450, "overlap_threshold": 1},
            [{0, 1, 2, 3, 4, 5}],
            [{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4}],
        ),
        (
            {"d_to_centroid": 450, "max_edge_size": 450, "overlap_threshold": 0.9},
            [{0, 1, 2, 3, 4}],
            [{0, 1, 2, 3, 4}],
        ),
        (
            {"d_to_centroid": 220, "max_edge_size": 450, "overlap_threshold": 1},
            [{1, 2, 3, 4, 5}],
            [{1, 2, 3, 4, 5}, {1, 4}],
        ),
    ),
)
def test_sample_subgraphs(
    simple_cells_float, sample_config, samples_indices, expected_indices
):
    """test sample subgraphs function: overlap, d_to_centroid"""
    random.seed(1)
    simple_cells_float = simple_cells_float.astype(float)
    new_adj, new_feats, updated_samples_indices = sample_subgraphs_from_cells_df(
        simple_cells_float, samples_indices, **sample_config
    )

    assert updated_samples_indices == expected_indices
    if new_adj is not None:
        sampled_indices = samples_indices[-1]
        assert (
            simple_cells_float.loc[sampled_indices]
            .reset_index(drop=True)
            .equals(new_feats)
        )

        centroids = torch.tensor(
            [
                (simple_cells_float.loc[i, "x"], simple_cells_float.loc[i, "y"])
                for i in simple_cells_float.index
            ]
        )
        sampled_dists = torch.cdist(centroids, centroids, p=2)

        assert sampled_dists.max(0).values.min() < sample_config["d_to_centroid"]
