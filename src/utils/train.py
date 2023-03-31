"""lighning training script"""
import logging
from typing import Any, Mapping, Optional

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback

log = logging.getLogger(__name__)


def model_training(
    model: pl.LightningModule,
    data_input: pl.LightningDataModule,
    trainer_params: Mapping[str, Any],
    logger: MLFlowLogger,
    tune_callback: Optional[TuneReportCallback] = None,
) -> pl.LightningModule:
    """General function for model training.
    Takes either a model input table or data module for the data.
    Args:
        data_input: Data module for generating train/test or model input dataframe
        model_object_path: Path to model to load
        trainer_params: Keyword args to be passed to training
        logger: ml flow logger
    Returns:
        Trained model.
    """
    early_stopping = EarlyStopping(
        monitor="train_loss",
        mode="min",
        verbose=False,
        patience=trainer_params["early_stopping_patience"],
    )
    lr_logger = LearningRateMonitor()

    callbacks = [
        lr_logger,
        early_stopping,
    ]
    if tune_callback:
        callbacks += [tune_callback]

    log.info("Starting training.")
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=25,
        max_epochs=trainer_params.get("max_epochs", -1),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else None,
        accumulate_grad_batches=trainer_params["accumulate_grad_batches"],
        enable_checkpointing=False,
    )
    trainer.fit(model=model, datamodule=data_input)
    log.info("Finished training.")
    mlflow.pytorch.log_model(model, "model")

    return model
