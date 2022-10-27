"""Tests for `torch_tools.models._adaptive_pools_2d`."""
import pytest

from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d

from torch_tools.models._adaptive_pools_2d import get_adaptive_pool
from torch_tools.models._adaptive_pools_2d import _ConcatMaxAvgPool2d


def test_get_adaptive_pool_option_argument_type():
    """Test `get_adaptive_pool` `option` argument type."""
    # Should work with string
    _ = get_adaptive_pool(option="max")
    _ = get_adaptive_pool(option="avg")
    _ = get_adaptive_pool(option="avg-max-concat")

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = get_adaptive_pool(option=1)
    with pytest.raises(TypeError):
        _ = get_adaptive_pool(option=1.0)


def test_get_adaptive_pool_option_values():
    """Test `get_adaptive_pool `option` values."""
    # Should work with the three options:
    _ = get_adaptive_pool(option="max")
    _ = get_adaptive_pool(option="avg")
    _ = get_adaptive_pool(option="avg-max-concat")

    # Should break with non-accepted options
    with pytest.raises(RuntimeError):
        _ = get_adaptive_pool(option="Radagast the Brown.")
    with pytest.raises(RuntimeError):
        _ = get_adaptive_pool(option="Gaffer Gamgee.")


def test_get_adaptive_pool_return_types():
    """Test the types return by `get_adaptive_pool`."""
    assert isinstance(get_adaptive_pool("avg"), AdaptiveAvgPool2d)

    assert isinstance(get_adaptive_pool("max"), AdaptiveMaxPool2d)

    assert isinstance(get_adaptive_pool("avg-max-concat"), _ConcatMaxAvgPool2d)
