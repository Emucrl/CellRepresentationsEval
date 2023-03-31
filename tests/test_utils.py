"""test utils utils"""
import pytest
from torch_geometric.nn import DeepGraphInfomax

from src.utils.pyg_model_zoo import GCNEncoder
from src.utils.utils import load_obj


@pytest.mark.parametrize(
    ("target", "expected"),
    (
        ("min", min),
        ("src.utils.pyg_model_zoo.GCNEncoder", GCNEncoder),
        ("torch_geometric.nn.DeepGraphInfomax", DeepGraphInfomax),
    ),
)
def test_load_object(target, expected) -> None:
    """test load_object"""
    loaded = load_obj(target)
    assert loaded == expected
