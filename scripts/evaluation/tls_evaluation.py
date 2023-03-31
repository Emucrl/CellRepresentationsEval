"""TLS Analysis pipeline"""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.experiments_nodes.tls_detection import generate_tls_masks, tls_detection
from src.utils import classification_results_utils, utils

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="defaults",
)
def pipeline(cfg: DictConfig) -> None:
    """Generate tls masks and performs cell level binary classification on embeddings

    The positive cells are defined with cfg.scripts.eval.bcells_query, which is applied as a pandas query.
    The data we used includes a cd20 expression column that we leveraged to identify positive cells.
    DBSCAN is applied on cells that meets the query to identify dense tls-like regions
    Masking parameters (min number of cells/min mask size etc can be controlled from conf/scripts/evaluations)

    Outputs will be stored in hydra_logs run folder
    """

    n_iterations = cfg.scripts.evaluations.n_iterations
    generate_tls_masks.generate_masks(cfg)
    for edge_size in [50, 30]:
        OmegaConf.update(cfg, "graph_creation.graphs.max_edge_size", edge_size)
        output_folder = Path(
            f"{cfg.paths.evaluation.tls_outputs}/outputs/edge_size={edge_size}"
        )
        utils.generate_test_sets_tls(cfg, output_folder, n_iterations)
        for iteration in tqdm(range(n_iterations)):
            log.info("Iteration %i", iteration)
            tls_detection.tls_classification_pipeline(cfg, output_folder, iteration)
            classification_results_utils.plot_results(
                output_folder / "classification_results.parquet"
            )


if __name__ == "__main__":
    pipeline()  # pylint: disable=E1120
