"""tests related to pyg"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, core, initialize
from torch_geometric.nn import DeepGraphInfomax

from src.data_processing_nodes.dgi_training import train_dgi
from src.data_processing_nodes.embedding_generation import generate_embedding_exp
from src.utils import graph_dataset
from src.utils.graph_utils import adj_feat_to_data, generate_graph_from_csv
from src.utils.pyg_model_zoo import GCNEncoder
from src.utils.pyg_utils import (
    embed_from_encoder,
    embed_from_full_model,
    features_permutation_corruption,
    identity_corruption,
    sigmoid_summary,
)

core.global_hydra.GlobalHydra.instance().clear()
initialize(version_base=None, config_path="../conf")


@pytest.fixture(name="data_with_weight")
def fixture_data_with_weight():
    """simple input"""
    cells = pd.DataFrame(
        {
            "x": [0, 0, 299, 299, 150],
            "y": [0, 299, 0, 299, 150],
            "feature": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    adj, features = generate_graph_from_csv(cells, 300)
    return adj_feat_to_data(adj, features, ["feature", "deg"])


@pytest.fixture(name="data_no_weight")
def fixture_data_no_weight():
    """simple input with no edge weight"""
    cells = pd.DataFrame(
        {
            "x": [0, 0, 299, 299, 150],
            "y": [0, 299, 0, 299, 150],
            "feature": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    adj, features = generate_graph_from_csv(cells, 300)
    adj = (adj != 0).astype(float)
    data = adj_feat_to_data(adj, features, ["feature", "deg"])
    return data


def test_edge_weight_gcn(data_with_weight, data_no_weight):
    """that that the same input with or without edge weight yield different outputs"""
    model = GCNEncoder(2, 2, 1, 0)

    emb_weight = model(
        data_with_weight.x,
        data_with_weight.edge_index,
        data_with_weight.edge_weight,
    ).detach()
    emb_no_weight = model(
        data_no_weight.x,
        data_with_weight.edge_index,
        data_no_weight.edge_weight,
    ).detach()
    assert np.array_equal(np.array(data_with_weight.x), np.array(data_no_weight.x))
    assert np.array_equal(
        np.array(data_no_weight.edge_index), np.array(data_with_weight.edge_index)
    )
    assert not np.array_equal(np.array(emb_no_weight), np.array(emb_weight))


def test_dropout_gcn(data_with_weight):
    """test that dropout is correctly applied during training but not eval"""
    model = GCNEncoder(2, 2, 2, 0.5)
    model.train()
    first = model(
        data_with_weight.x,
        data_with_weight.edge_index,
        data_with_weight.edge_weight,
    ).detach()
    second = model(
        data_with_weight.x,
        data_with_weight.edge_index,
        data_with_weight.edge_weight,
    ).detach()

    model.eval()

    third = model(
        data_with_weight.x,
        data_with_weight.edge_index,
        data_with_weight.edge_weight,
    ).detach()

    fourth = model(
        data_with_weight.x,
        data_with_weight.edge_index,
        data_with_weight.edge_weight,
    ).detach()
    assert not np.array_equal(np.array(first), np.array(second))
    assert np.array_equal(np.array(third), np.array(fourth))


def test_permutation(data_with_weight):
    """test that permutation changes when applied multiple times"""
    a_feats, a_edge_index, a_edge_weights = features_permutation_corruption(
        data_with_weight.x, data_with_weight.edge_index, data_with_weight.edge_weight
    )

    b_feats, b_edge_index, b_edge_weights = features_permutation_corruption(
        data_with_weight.x, data_with_weight.edge_index, data_with_weight.edge_weight
    )

    assert not np.array_equal(np.array(data_with_weight.x), np.array(a_feats))
    assert np.array_equal(np.array(a_edge_index), np.array(data_with_weight.edge_index))
    assert np.array_equal(
        np.array(a_edge_weights), np.array(data_with_weight.edge_weight)
    )

    assert not np.array_equal(np.array(b_feats), np.array(a_feats))
    assert np.array_equal(np.array(a_edge_index), np.array(b_edge_index))
    assert np.array_equal(np.array(a_edge_weights), np.array(b_edge_weights))
    assert not np.array_equal(np.array(data_with_weight.x), np.array(b_feats))

    a_feats, a_edge_index, a_edge_weights = identity_corruption(
        data_with_weight.x, data_with_weight.edge_index, data_with_weight.edge_weight
    )
    assert np.array_equal(a_feats, data_with_weight.x)
    assert np.array_equal(a_edge_index, data_with_weight.edge_index)
    assert np.array_equal(a_edge_weights, data_with_weight.edge_weight)


def test_embedding(data_with_weight):
    """test embedding from full model or just encoder and that corruption don't apply at eval"""
    encoder = GCNEncoder(2, 2, 1, 0)
    model = DeepGraphInfomax(
        hidden_channels=2,
        encoder=encoder,
        summary=lambda z, *args, **kwargs: z.mean(dim=0),
        corruption=features_permutation_corruption,
    )
    model.eval()
    model2 = DeepGraphInfomax(
        hidden_channels=2,
        encoder=encoder,
        summary=lambda z, *args, **kwargs: z.mean(dim=0),
        corruption=identity_corruption,
    )
    model2.eval()
    emb1 = embed_from_encoder(encoder, data_with_weight)
    emb2 = embed_from_full_model(model, data_with_weight)
    emb3 = embed_from_encoder(model.encoder, data_with_weight)

    emb4 = embed_from_full_model(model2, data_with_weight)
    emb5 = embed_from_encoder(model2.encoder, data_with_weight)

    assert np.all(
        [
            p1.data.ne(p2.data).sum() == 0
            for p1, p2 in zip(model.encoder.parameters(), model2.encoder.parameters())
        ]
    )
    assert np.array_equal(emb1, emb2)
    assert np.array_equal(emb1, emb3)
    assert np.array_equal(emb1, emb4)
    assert np.array_equal(emb1, emb5)


def test_embedding_script(tmp_path):
    """test traing_dgi and generate_embedding_exp"""

    exp_name = "defaults"
    cfg = compose(
        exp_name,
        overrides=[
            "++dgi.training_params.trainer_params.max_epochs=2",
            "++dgi.mlflow.experiment_name=default",
        ],
    )
    cfg["dgi"]["training_params"]["features_list"] = [
        "Area",
        "Circ",
        "deg",
    ]

    exp_cfg = cfg["dgi"]
    exp_cfg["mlflow"]["tracking_uri"] = str(tmp_path)
    exp_cfg["training_params"]["trainer_params"]["max_epochs"] = 2
    exp_cfg["training_params"]["model_params"] = {
        "hid_units": 2,
        "n_layers": 1,
        "dropout": 0.1,
    }

    training_dataset = graph_dataset.GraphDataset(
        Path("tests/test_data/graphs"),
        features_list=exp_cfg["training_params"]["features_list"],
    )

    train_dgi(config=cfg, datapath=Path("tests/test_data/graphs"))
    generate_embedding_exp(
        exp_embeddings_path=tmp_path / "embeddings",
        mlflow_path=str(tmp_path),
        embedding_generation_config=cfg["embedding_generation"],
        input_data_path=Path("tests/test_data/graphs"),
        features_list=exp_cfg["training_params"]["features_list"],
    )
    assert len(list((tmp_path / "embeddings").iterdir())) == len(training_dataset)


def test_sigmoid_summary():
    """test sigmoid summary"""
    tensor = torch.ones((10, 2))
    summary = sigmoid_summary(tensor)
    assert summary.shape == torch.Size([2])
    assert torch.allclose(torch.sigmoid(torch.ones((1, 2))), summary)

    tensor = torch.zeros((2, 4))
    summary = sigmoid_summary(tensor)
    assert summary.shape == torch.Size([4])
    assert torch.allclose(torch.sigmoid(torch.zeros((1, 4))), summary)

    tensor = torch.rand((200, 4))
    summary = sigmoid_summary(tensor)
    assert summary.shape == torch.Size([4])
    assert torch.allclose(torch.sigmoid(0.5 * torch.ones((1, 4))), summary, rtol=1e-1)
