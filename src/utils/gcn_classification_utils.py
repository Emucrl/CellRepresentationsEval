"""GCN classification utils"""
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.loader import DataLoader

from src.utils import (
    classification_results_utils,
    gcn_classification_utils,
    pyg_model_zoo,
)
from src.utils.graph_dataset import GraphDataset


def train_gcn(
    cfg: Mapping,
    model: pl.LightningModule,
    train_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
) -> Tuple[pl.LightningModule, pl.Trainer]:
    """train model on traing_dataset and comput test metrics on test_dataset
    Args:
        cfg (Mapping)
        model (pl.LightningModule)
        train_dataset (Dataset)
        test_dataset (Optional[Dataset], optional)

    Returns:
        Tuple[pl.LightningModule, pl.Trainer]: trained model and corresponding pl.trainer
    """
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        verbose=False,
        patience=cfg["early_stopping_patience"],
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=False,
        monitor="val_auc",
        mode="max",
        dirpath=cfg["checkpoint_path"],
        filename="best_model",
    )
    callbacks = [early_stopping, checkpoint_callback]
    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=cfg["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else None,
        enable_checkpointing=True,
    )
    train_dataset, validation_dataset = torch.utils.data.random_split(
        train_dataset,
        [
            train_size := int(0.8 * len(train_dataset)),
            len(train_dataset) - train_size,
        ],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=32,
        num_workers=8,
    )
    trainer.fit(
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
        model=model,
    )
    if test_dataset:
        test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=8)
        trainer.test(
            ckpt_path="best",
            dataloaders=test_dataloader,
        )
    return model, trainer


def train_predict(
    model: pyg_model_zoo.GCNClassifier,
    train_dataset: Dataset,
    test_dataset: Dataset,
    cfg: Mapping,
) -> dict[str, Any]:
    """train gcn and perform inference on test set to store both metrics and predictions/labels"""
    model, trainer = train_gcn(
        cfg,
        model,
        train_dataset,
        test_dataset,
    )
    dgi_outputs = trainer.predict(
        ckpt_path="best",
        dataloaders=DataLoader(test_dataset, num_workers=8),
    )
    dgi_preds = torch.concat([output["preds"] for output in dgi_outputs])
    dgi_gts = torch.concat([output["gts"] for output in dgi_outputs])
    results = model.results_function(dgi_preds, dgi_gts)
    return results


def generate_dataset(
    training_patients: List[str],
    test_patients: List[str],
    cfg: DictConfig,
    dataset_class: Callable,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """generate train and test datasets
    Leverage Concatdataset to produce a single dataset from multiple data folders"""
    dir_graphs = Path(cfg.paths.dataset.patient_graphs)

    folders = list(folder for folder in dir_graphs.rglob("*/*") if not folder.is_file())
    train_dataset: torch.utils.data.Dataset = ConcatDataset(
        dataset_class(
            dir_graphs / folder,
            training_patients,
            cfg=cfg,
        )
        for folder in folders
    )

    test_dataset: torch.utils.data.Dataset = ConcatDataset(
        dataset_class(
            dir_graphs / folder,
            test_patients,
            cfg=cfg,
        )
        for folder in folders
    )
    return train_dataset, test_dataset


def node_level_binary_gcn(
    training_patients: List[str],
    test_patients: List[str],
    cfg: DictConfig,
    dataset_class: Callable,
) -> dict[str, List]:
    """Train supervised GCN for node-level prediction and performs inference

    Returns:
        dict[str, list]: mapping with predictions, labels and metrics"""
    model = pyg_model_zoo.GCNClassifier(
        pyg_model_zoo.GCNNodeClassif(
            len(cfg.dgi.training_params.features_list),
            cfg.dgi.training_params.model_params.hid_units,
            cfg.dgi.training_params.model_params.n_layers,
            cfg.dgi.training_params.model_params.dropout,
            2,
        ),
        cfg.scripts.evaluations.cell_level_experiments.gcn_optim_params,
        classification_results_utils.get_binary_classif_results,
    )
    train_dataset, test_dataset = generate_dataset(
        training_patients, test_patients, cfg, dataset_class
    )

    return gcn_classification_utils.train_predict(
        model,
        train_dataset,
        test_dataset,
        cfg.scripts.evaluations.cell_level_experiments.gcn_optim_params,
    )


def generate_multiclass_dataset(
    graph_dataset: Callable,
    training_patients: List[str],
    test_patients: List[str],
    cfg: DictConfig,
    gts_df: pd.DataFrame,
    classes: List[str],
) -> Tuple[GraphDataset, GraphDataset]:
    """Generates graph_datasets from list of training and test patients"""
    dir_graphs = Path(cfg.paths.dataset.patient_graphs)

    train_dataset = graph_dataset(
        dir_graphs,
        features_list=cfg.dgi.training_params.features_list,
        patients=training_patients,
        gts_df=gts_df,
        classes=classes,
    )

    test_dataset = graph_dataset(
        dir_graphs,
        features_list=cfg.dgi.training_params.features_list,
        patients=test_patients,
        gts_df=gts_df,
        classes=classes,
    )
    return train_dataset, test_dataset


def graph_level_multiclass_gcn(
    graph_dataset: Callable,
    training_patients: List[str],
    test_patients: List[str],
    cfg: DictConfig,
    gts_df: pd.DataFrame,
    classes: List[str],
) -> dict[str, list]:
    """Train supervised GCN for graph-level prediction and performs inference

    Returns:
        dict[str, list]: mapping with predictions, labels and metrics"""
    train_dataset, test_dataset = generate_multiclass_dataset(
        graph_dataset,
        training_patients,
        test_patients,
        cfg,
        gts_df=gts_df,
        classes=classes,
    )
    model = pyg_model_zoo.GCNClassifier(
        pyg_model_zoo.GCNGraphClassif(
            len(cfg.dgi.training_params.features_list),
            cfg.dgi.training_params.model_params.hid_units,
            cfg.dgi.training_params.model_params.n_layers,
            cfg.dgi.training_params.model_params.dropout,
            len(classes),
        ),
        cfg.scripts.evaluations.graph_level_experiments.gcn_optim_params,
        results_function=partial(
            classification_results_utils.get_multiclass_classif_results,
            n_classes=len(classes),
        ),
    )

    return gcn_classification_utils.train_predict(
        model,
        train_dataset,
        test_dataset,
        cfg.scripts.evaluations.graph_level_experiments.gcn_optim_params,
    )
