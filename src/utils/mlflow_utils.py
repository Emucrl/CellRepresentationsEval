"""utils functions related to MLFlow and Training config"""
import logging
from typing import Callable

import mlflow
import torch_geometric
from mlflow.entities import Run
from pytorch_lightning.loggers import MLFlowLogger

log = logging.getLogger(__name__)


def get_best_run(
    exp_id: str,
    mlflow_path: str,
    metric: str = "loss",
    mode: Callable = min,
    only_logged_models: bool = False,
) -> Run:
    """get the best run the mlflow_path exp_id according to metric and mode (min max)

    Args:
        exp_id (str): mlflow experiment id
        mlflow_path (str): path to the mlruns repo
        metric (str, optional): _description_. Defaults to "loss".
        mode Function: min or max
        only_logged_models: select only among runs with a logged model to load it downstream
    Returns:
       mlflow.projects.Run
    """
    mlflow.set_tracking_uri(mlflow_path)
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs([exp_id], "")
    if only_logged_models:
        runs = [run for run in runs if run.data.tags.get("mlflow.log-model.history")]
    metrics = []
    for run in runs:
        metrics += [run.data.metrics.get(metric)]
    best_score = mode([metric for metric in metrics if metric is not None])
    best_run = runs[metrics.index(best_score)]
    return best_run


def get_best_model(
    exp_name: str, mlflow_path: str, metric: str = "loss", mode: Callable = min
) -> torch_geometric.nn.DeepGraphInfomax:
    """load model in the experiment exp_name of the registry mlflow_path
    with the mode (min or max) value of metric
    return the eval() version of the model (no dropout)

    Args:
        exp_name (str): mlflow experiment name
        mlflow_path (str): path to the mlruns repo
        metric (str, optional): _description_. Defaults to "loss".
        mode Function: min or max
    Returns:
        torch.nn.module: torch model
    """
    exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    best_run = get_best_run(exp_id, mlflow_path, metric, mode, only_logged_models=True)
    best_model = mlflow.pytorch.load_model(best_run.info.artifact_uri + "/model")
    best_model.eval()
    return best_model


def create_experiment(
    mlflow_path: str,
    experiment_name: str,
) -> MLFlowLogger:
    """Creates an MLflow experiment.
    Args:
        mlflow_base_name: Base of experiment name in remote server
        mlflow_path: Path where the mlruns folder will be stored on the machine.
        experiment_name: Name of experiment
        tags: Dictionary of tags to add to the run logs
    Returns:
        MLFlowLogger:  MLflow logger for the experiment.
    """
    mlflow.set_tracking_uri(mlflow_path)
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=mlflow_path)
    run_id = mlf_logger.run_id
    mlflow.start_run(run_id, nested=True)
    log.info("Experiment name: %s", experiment_name)
    log.info("Experiment tracking URI: %s", mlflow.tracking.get_tracking_uri())
    return mlf_logger
