"""
pyg models
"""
from typing import Any, Callable, Dict, List, Mapping

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.typing import Adj


class GCNEncoder(nn.Module):
    """Encoder with GCN conv layers"""

    def __init__(
        self, in_channels: int, hidden_channels: int, n_layers: int, dropout: float
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(in_channels, hidden_channels))
        self.activation_layers = nn.ModuleList()
        self.activation_layers.append(nn.PReLU(hidden_channels))
        self.dropout = dropout

        for _ in range(max(0, n_layers - 1)):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.activation_layers.append(nn.PReLU(hidden_channels))

    def forward(self, features: Tensor, edge_index: Adj, edge_weight: Tensor) -> Tensor:
        """forward"""
        features = F.dropout(features, p=self.dropout, training=self.training)
        for convlayer, activationlayer in zip(self.conv_layers, self.activation_layers):
            features = convlayer(features, edge_index, edge_weight=edge_weight)
            features = activationlayer(features)
        return features


class GCNNodeClassif(nn.Module):
    """node classifier"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_layers: int,
        dropout: float,
        nclasses: int,
    ) -> None:
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, n_layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, nclasses),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """forward"""
        out = self.encoder(data.x, data.edge_index, data.edge_weight)
        out = self.classifier(out)
        return out


class GCNGraphClassif(nn.Module):
    """Graph classifier"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_layers: int,
        dropout: float,
        nclasses: int,
    ) -> None:
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, n_layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, nclasses),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """forward"""
        out = self.encoder(data.x, data.edge_index, data.edge_weight)
        out = global_mean_pool(out, data.batch)
        out = self.classifier(out)
        return out


class GCNClassifier(
    pl.LightningModule
):  # pylint: disable=too-many-ancestors, too-many-instance-attributes
    """lightning model to wrap around a pytorch_geometric model"""

    def __init__(
        self,
        model: nn.Module,
        optimization_params: Dict | DictConfig,
        results_function: Callable,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = float(optimization_params["lr"])
        self.trainer: pl.Trainer
        self.loss = nn.CrossEntropyLoss()
        self.results_function = results_function

    def forward(self, batch: Batch) -> Tensor:
        return self.model(batch)

    def training_step(self, batch: Batch, _: int) -> Mapping[str, torch.Tensor]:  # type: ignore
        """training step by calling torch-geometric self.loss

        Args:
            data: torch_geometrics
            _ : batch index, not used

        Returns:
            step loss
        """
        out = self(batch)
        loss = self.loss(out, batch.y)
        return {
            "loss": loss,
        }

    def training_epoch_end(self, outputs: List[Tensor | Dict[str, Any]]) -> None:  # type: ignore
        """Computes average training epoch loss.
        Args:
            outputs: Outputs after every epoch end.
        """
        # for some reason, the step loss is divides by accumulate btwn training_step and epoch_end
        # multiply here to log the right value
        avg_loss = (
            self.trainer.accumulate_grad_batches
            * sum(x["loss"].item() for x in outputs)  # type: ignore
            / len(outputs)
        )
        self.log("train_loss", avg_loss, logger=False, prog_bar=True, sync_dist=True)

    def predict_step(self, batch: Batch, _: Any) -> Dict:  # type: ignore
        return {
            "preds": self(batch),
            "gts": batch.y,
        }

    def test_step(self, batch: Batch, _: Any) -> Dict:
        preds = self(batch)
        return {
            "loss": self.loss(preds, batch.y).item(),
            "preds": preds,
            "gts": batch.y,
        }

    def validation_step(self, batch: Batch, _: Any) -> Dict[str, torch.Tensor]:  # type: ignore
        """validation step by calling torch-geometric self.loss

        Args:
            data: torch_geometrics
            _ : batch index, not used

        Returns:
            step loss
        """
        out = self.model(batch)
        loss = self.loss(out, batch.y)
        return {"loss": loss, "output": out, "gt": batch.y}

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:  # type: ignore
        avg_loss = torch.tensor([output["loss"] for output in outputs]).mean()
        preds = torch.concat([output["preds"] for output in outputs])
        gts = torch.concat([output["gts"] for output in outputs])
        results = self.results_function(preds, gts)
        results["test_loss"] = avg_loss
        self.log_dict(
            {
                key: value
                for key, value in results.items()
                if key not in ["probas", "labels"]
            }
        )

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:  # type: ignore
        """Computes average validation loss.
        Args:
            outputs: Outputs after every epoch end.
        """
        # for some reason, the step loss is divides by accumulate btwn validation_step and epoch_end
        # multiply here to log the right value
        avg_loss = (
            self.trainer.accumulate_grad_batches
            * sum(x["loss"].item() for x in outputs)  # type: ignore
            / len(outputs)
        )
        preds = torch.concat([output["output"] for output in outputs])
        gts = torch.concat([output["gt"] for output in outputs])
        results = self.results_function(preds, gts)
        self.log("val_loss", avg_loss, logger=True, prog_bar=True, sync_dist=True)
        self.log("val_auc", results["auc"], logger=True, prog_bar=True, sync_dist=True)
        self.log("val_f1", results["f1"], logger=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer:
        """Initializes the optimizer and learning rate scheduler.
        Returns:
            Tuple: Initialized optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
