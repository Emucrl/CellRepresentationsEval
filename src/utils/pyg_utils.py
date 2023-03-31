"""pyg utils"""
from typing import Any, Tuple

import numpy as np
import torch
import torch_geometric
from torch import nn
from torch_geometric.data.data import BaseData


def embed_from_full_model(
    model: torch_geometric.nn.DeepGraphInfomax, data: BaseData
) -> np.ndarray:
    """embeddings from full model (like dgi with encoder+prediction head)

    Args:
        model (nn.Module)
        data (torch_geometric.data.Data)

    Returns:
        np.array embedded features
    """
    return embed_from_encoder(model.encoder, data)


def embed_from_encoder(model: nn.Module, data: BaseData) -> np.ndarray:
    """embedding using only an encoder"""
    model.eval()
    with torch.no_grad():
        model.cpu()
        data.cpu()
        embeds = model.forward(
            data.x,
            data.edge_index,
            data.edge_weight,
        )
        embeds = embeds.numpy()
    return embeds


def features_permutation_corruption(
    features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """randomly assign feature vectors to nodes as per DGI paper"""
    return features[torch.randperm(features.size(0))], edge_index, edge_weight


def identity_corruption(
    features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """identity"""
    return features, edge_index, edge_weight


def sigmoid_summary(tensor: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
    """Avg the tensor along the cells dimensions
    tensor: NxD, output: 1*D
    """
    return torch.sigmoid(tensor.mean(dim=0))
