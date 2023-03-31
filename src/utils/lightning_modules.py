"""lightning modules"""
import os
from typing import Any, Dict, List, Mapping, Tuple

import pytorch_lightning as pl
import torch
import torch_geometric
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Adj

from src.utils.graph_dataset import GraphDataset


class Model(
    pl.LightningModule
):  # pylint: disable=too-many-ancestors, too-many-instance-attributes
    """lightning model to wrap around a pytorch_geometric model"""

    def __init__(
        self,
        model: torch_geometric.nn.DeepGraphInfomax,
        optimization_params: Dict | DictConfig,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = float(optimization_params["lr"])
        self.step_size = int(optimization_params["lr_scheduler_step_size"])
        self.gamma = float(optimization_params["lr_scheduler_gamma"])
        self.weight_decay = float(optimization_params["l2_coeff"])
        self.optimizer, self.scheduler = self.configure_optimizers()
        self.logger: MLFlowLogger
        self.trainer: pl.Trainer

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: Tensor) -> Tensor:  # type: ignore
        return self.model(x, edge_index, edge_weight)

    def training_step(self, batch: Batch, _: int) -> Mapping[str, torch.Tensor]:  # type: ignore
        """training step by calling torch-geometric self.loss

        Args:
            data: torch_geometrics
            _ : batch index, not used

        Returns:
            Mapping: loss for the step
        """
        pos_z, neg_z, summary = self.forward(
            batch.x, batch.edge_index, batch.edge_weight
        )
        loss = self.model.loss(pos_z, neg_z, summary)
        return {
            "loss": loss,
        }

    def training_epoch_end(self, outputs: List[Tensor | Dict[str, Any]]) -> None:
        """Computes average validation loss.
        Args:
            outputs: Outputs after every epoch end.
        """
        # for some reason, the step loss is divides by accumulate btwn training_step and epoch_end
        # multiply here to log the right value
        avg_loss = (
            self.trainer.accumulate_grad_batches
            * sum(x["loss"].cpu().item() for x in outputs)  # type: ignore
            / len(outputs)
        )
        self.log("train_loss", avg_loss, logger=False, prog_bar=True, sync_dist=True)
        self.logger.log_metrics({"train_loss": avg_loss}, step=self.current_epoch)

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer],
        List[Dict[str, torch.optim.lr_scheduler._LRScheduler]],
    ]:
        """Initializes the optimizer and learning rate scheduler.
        Returns:
            Tuple: Initialized optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.step_size, gamma=self.gamma
            ),
        }
        self.optimizer = optimizer  # type: ignore
        self.scheduler = scheduler  # type: ignore

        return [optimizer], [scheduler]  # type: ignore


class GraphDataModule(pl.LightningDataModule):
    """Data loader for ellipse models."""

    def __init__(self, graph_dataset: GraphDataset, batch_size: int):
        super().__init__()
        self.dataset = graph_dataset
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
            persistent_workers=True,
        )
        return train_loader
