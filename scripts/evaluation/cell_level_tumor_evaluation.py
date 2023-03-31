"""TLS Analysis pipeline"""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.experiments_nodes.tumor_detection import tumor_detection
from src.utils import classification_results_utils, utils

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="defaults",
)
def pipeline(cfg: DictConfig) -> None:
    """Generate tls masks and performs cell level binary classification on embeddings

    This pipeline assumes that the cells csv files contains a tumour_mask columns
    Changes can be made to cell_level_classification_utils.get_tumor_gt for tumor gt generation
    DBSCAN is applied on cells with high values of this column to identify tls-like regions

    Outputs will be stored in hydra_logs run folder
    """

    n_iterations = cfg.scripts.evaluations.n_iterations

    for edge_size in [50, 30]:
        OmegaConf.update(cfg, "graph_creation.graphs.max_edge_size", edge_size)
        output_folder = Path(
            f"{cfg.paths.evaluation.tumor_outputs}/outputs/edge_size={edge_size}"
        )
        utils.generate_test_sets(cfg, output_folder, n_iterations)
        for iteration in tqdm(range(n_iterations)):
            log.info("Iteration %i", iteration)
            tumor_detection.tumor_classification_pipeline(cfg, output_folder, iteration)
            classification_results_utils.plot_results(
                output_folder / "classification_results.parquet"
            )


if __name__ == "__main__":
    pipeline()  # pylint: disable=E1120
