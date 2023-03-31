"""full pipeline"""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.data_processing_nodes import (
    dgi_training,
    embedding_generation,
    message_passing_generation,
    patient_graph_generation,
    subgraphs,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="defaults",
)
def pipeline(cfg: DictConfig) -> None:
    """Embeddings generation pipeline
    generate graphs from cells files, train DGI and embeds graphs
    with both DGI and message passing

    run with --multirun to trigger for sweeps defined in hydra.sweeper.params

    Args:
        cfg (DictConfig): Hydra cfg loaded by decorator

    """
    log.info("********** RUNNING TRAIN PIPELINE.PY ***********")

    for dataset in cfg["scripts"]["data_processing"]["graph_generation_datasets"]:
        log.info("Start generating graphs for %s", dataset)
        patient_graph_generation.generate_all_graphs(
            source_data_path=Path(cfg["paths"][dataset]["cells"]),
            target_path=Path(cfg["paths"][dataset]["patient_graphs"]),
            max_edge_size=cfg["graph_creation"]["graphs"]["max_edge_size"],
            n_splits=cfg["graph_creation"][dataset]["n_splits"],
        )
        log.info("Generated patient graphs for %s", dataset)

    if cfg["scripts"]["data_processing"]["train"]:
        for training_dataset in cfg["scripts"]["data_processing"]["training_dataset"]:
            log.info("Training DGI on dataset %s", training_dataset)

            subgraphs.generate_subgraphs_all_patients(
                Path(cfg["paths"][training_dataset]["cells"]),
                subgraphs_config=cfg["graph_creation"]["subgraphs"],
                save_path=Path(cfg["paths"][training_dataset]["subgraphs"]),
            )
            log.info("Generated subgraphs")

            dgi_training.train_dgi(
                cfg, datapath=Path(cfg["paths"][training_dataset]["subgraphs"])
            )

    for dataset in cfg["scripts"]["data_processing"]["embeddings_generation_datasets"]:
        embedding_generation.generate_embedding_exp(
            input_data_path=Path(cfg["paths"][dataset]["patient_graphs"]),
            exp_embeddings_path=Path(cfg["paths"][dataset]["embeddings"]),
            mlflow_path=cfg["dgi"]["mlflow"]["tracking_uri"],
            embedding_generation_config=cfg["embedding_generation"],
            features_list=cfg["dgi"]["training_params"]["features_list"],
        )
        log.info("Embeddings generated for %s", dataset)

        message_passing_generation.message_passing_generation_node(
            Path(cfg["paths"][dataset]["patient_graphs"]),
            Path(cfg["paths"][dataset]["message_passing"]),
            features_list=cfg["dgi"]["training_params"]["features_list"],
        )


if __name__ == "__main__":
    pipeline()  # pylint: disable=E1120
