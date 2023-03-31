"""scripts to generate embeddings"""
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import mlflow
from tqdm import tqdm

from src.utils import graph_dataset, mlflow_utils, pyg_utils, utils


def generate_embedding_exp(
    exp_embeddings_path: Path,
    mlflow_path: str,
    embedding_generation_config: Dict[str, Any],
    input_data_path: Path,
    features_list: List[str],
) -> None:
    """Load all graphs in input_data_path,
    Load the best model from exp mlflow_exp_id in the mlflow_path registry
    embed all loaded graphs and store the numpy array in exp_embeddings_path/mlflow_exp_id

    Inference is done on CPU: no time constraint but allows to infer for very large graphs

    Args:
        exp_embeddings_path (Path):  Folder to write embeddings
        mlflow_path (str)
        embedding_generation_config (Dict[str, Any]):
            - mlflow_exp_name
            - best_model_config:
                - metric
                - mode (str of "min" or "max")
        input_data_path (Path): Path to graphs to embed
        features_list (List[str]): list of features used for DGI training and to use for embedding
    """
    mlflow.set_tracking_uri(mlflow_path)

    logging.info(
        "######### LOAD MODEL FROM EXP %s #######",
        embedding_generation_config["mlflow_exp_name"],
    )

    model = mlflow_utils.get_best_model(
        embedding_generation_config["mlflow_exp_name"],
        mlflow_path=mlflow_path,
        metric=embedding_generation_config["best_model_config"]["metric"],
        mode=utils.load_obj(embedding_generation_config["best_model_config"]["mode"]),
    )

    patient_graphs_dataset = graph_dataset.GraphDataset(
        input_data_path,
        features_list=features_list,
    )

    for idx, data in tqdm(enumerate(iter(patient_graphs_dataset))):
        embedding = pyg_utils.embed_from_full_model(model.model, data)

        slide_num = "_".join(
            patient_graphs_dataset.graphs_paths[idx].name.split("_")[:-1]
        )
        print(slide_num)
        filename = (
            utils.swap_root_directory(
                input_data_path,
                exp_embeddings_path,
                patient_graphs_dataset.graphs_paths[idx],
            ).parent
            / f"{slide_num}_embeddings.p"
        )
        filename.parent.mkdir(exist_ok=True, parents=True)
        with open(
            filename,
            "wb",
        ) as file:
            pickle.dump(
                embedding,
                file,
            )
