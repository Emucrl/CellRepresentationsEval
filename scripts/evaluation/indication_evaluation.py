"""Indication prediction pipeline"""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.experiments_nodes.indication_detection import prediction_pipeline
from src.utils import classification_results_utils, utils

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="defaults",
)
def pipeline(cfg: DictConfig) -> None:
    """Performs roi level classification on indication

    Pipeline expects a pd.Dataframe with 'patient' and 'indication' column under gts_df

    We used the function utils.find_patient_indications that parses the cells filenames,
    as our dataset is stored with indication=XXX/datadump=XXX/PATIENTID_cells.csv

    One can replace this function by a simple read_csv if groundtruth is available

    cfg.paths.dataset.embeddings and cfg.paths.dataset.message_passing should be populated

    Outputs will be stored in hydra_logs run folder, plot are automatically updated each iteration
    """

    n_iterations = cfg.scripts.evaluations.n_iterations

    gts_df = utils.find_patient_indications(cfg)

    for edge_size in [50, 30]:
        OmegaConf.update(cfg, "graph_creation.graphs.max_edge_size", edge_size)
        output_folder = Path(
            f"{cfg.paths.evaluation.indication_outputs}/outputs/edge_size={edge_size}"
        )
        utils.generate_test_sets(cfg, output_folder, n_iterations)
        for iteration in tqdm(range(n_iterations)):
            log.info("Iteration %i", iteration)
            prediction_pipeline.indication_classification_pipeline(
                cfg, output_folder, iteration, gts_df
            )
            classification_results_utils.plot_results(
                output_folder / "indication_classification.parquet"
            )


if __name__ == "__main__":
    pipeline()  # pylint: disable=E1120
