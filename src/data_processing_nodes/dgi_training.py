"""dgi_training
"""
import random
from pathlib import Path

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch_geometric.nn import DeepGraphInfomax

from src.utils import (
    graph_dataset,
    lightning_modules,
    mlflow_utils,
    pyg_model_zoo,
    pyg_utils,
    train,
)

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
pl.seed_everything(seed=0)


def train_dgi(config: DictConfig, datapath: Path) -> None:
    """Train the pytorch-geometric model
    Training is performed using torch-lighning
    Training logged in mlflow

    Args:
        config (DictConfig): Hydra config with all experiments parameters
        datapath (Path): Path to the folder with training graphs
    """
    mlf_logger = mlflow_utils.create_experiment(
        mlflow_path=config["dgi"]["mlflow"]["tracking_uri"],
        experiment_name=config["dgi"]["mlflow"]["experiment_name"],
    )

    mlflow.log_param("features_list", config["dgi"]["training_params"]["features_list"])
    mlflow.log_params(dict(config["dgi"]["training_params"]["trainer_params"].items()))
    mlflow.log_params(
        dict(config["dgi"]["training_params"]["data_module_params"].items())
    )
    mlflow.log_params(
        dict(config["dgi"]["training_params"]["optimization_params"].items())
    )
    mlflow.log_params(dict(config["dgi"]["training_params"]["model_params"].items()))

    dataset = graph_dataset.GraphDataset(
        datapath,
        features_list=config["dgi"]["training_params"]["features_list"],
        limit_size=config["dgi"]["training_params"].get("limit_size", 100000),
    )

    model = DeepGraphInfomax(
        hidden_channels=config["dgi"]["training_params"]["model_params"]["hid_units"],
        encoder=pyg_model_zoo.GCNEncoder(
            len(config["dgi"]["training_params"]["features_list"]),
            config["dgi"]["training_params"]["model_params"]["hid_units"],
            config["dgi"]["training_params"]["model_params"]["n_layers"],
            config["dgi"]["training_params"]["model_params"]["dropout"],
        ),
        summary=pyg_utils.sigmoid_summary,
        corruption=pyg_utils.features_permutation_corruption,
    )
    model = lightning_modules.Model(
        model,
        optimization_params=config["dgi"]["training_params"]["optimization_params"],
    )
    datamodule = lightning_modules.GraphDataModule(
        dataset, config["dgi"]["training_params"]["data_module_params"]["batch_size"]
    )
    train.model_training(
        model,
        datamodule,
        config["dgi"]["training_params"]["trainer_params"],
        logger=mlf_logger,
    )
