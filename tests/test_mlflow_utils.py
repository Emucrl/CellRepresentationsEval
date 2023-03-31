"""test mlflow_utils functions"""
from src.utils import mlflow_utils


def test_get_best_run():
    """test select best run for different metrics and min/max"""
    run_min = mlflow_utils.get_best_run("0", "tests/test_data/mlruns", "1", min)
    assert run_min.info.run_uuid == "24d38bebe5d843cc9ff21032081d5db6"
    run_max = mlflow_utils.get_best_run("0", "tests/test_data/mlruns", "a", max)
    assert run_max.info.run_uuid == "c75980b397c3419aad305ce1b6317500"
