"""Tests for `torch_tools.models._adaptive_pools_2d`."""
import pytest

from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d

from torch_tools.models._adaptive_pools_2d import get_adaptive_pool
from torch_tools.models._adaptive_pools_2d import _ConcatMaxAvgPool2d


def test_get_adaptive_pool_option_argument_type():
    """Test `get_adaptive_pool` `option` argument type."""
    # Should work with string
    _ = get_adaptive_pool(option="max", output_size=(1, 1))
    _ = get_adaptive_pool(option="avg", output_size=(1, 1))
    _ = get_adaptive_pool(option="avg-max-concat", output_size=(1, 1))

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = get_adaptive_pool(option=1, output_size=(1, 1))
    with pytest.raises(TypeError):
        _ = get_adaptive_pool(option=1.0, output_size=(1, 1))


def test_get_adaptive_pool_option_values():
    """Test `get_adaptive_pool `option` values."""
    # Should work with the three options:
    _ = get_adaptive_pool(option="max", output_size=(1, 1))
    _ = get_adaptive_pool(option="avg", output_size=(1, 1))
    _ = get_adaptive_pool(option="avg-max-concat", output_size=(1, 1))

    # Should break with non-accepted options
    with pytest.raises(ValueError):
        _ = get_adaptive_pool(option="Radagast the Brown.", output_size=(1, 1))
    with pytest.raises(ValueError):
        _ = get_adaptive_pool(option="Gaffer Gamgee.", output_size=(1, 1))


def test_get_adaptive_pool_return_types():
    """Test the types return by `get_adaptive_pool`."""
    assert isinstance(
        get_adaptive_pool("avg", output_size=(1, 1)),
        AdaptiveAvgPool2d,
    )

    assert isinstance(
        get_adaptive_pool("max", output_size=(1, 1)),
        AdaptiveMaxPool2d,
    )

    assert isinstance(
        get_adaptive_pool("avg-max-concat", output_size=(1, 1)), _ConcatMaxAvgPool2d
    )


def test_get_adaptive_pool_output_size_type():
    """Test the type of the output_size arg of `get_adaptive_pool`.

    `output_size` argument should be a tuple.

    """
    # Should work with Tuple[int, int]
    _ = get_adaptive_pool(option="avg", output_size=(1, 1))

    # Should break with non-tuple
    with pytest.raises(TypeError):
        _ = get_adaptive_pool(option="avg", output_size=[1, 1])


def test_get_adaptive_pool_output_size_types():
    """Test types of elements in `output_size` arg of `get_adaptive_pool`.

    `output_size` argument should only contain integers.

    """
    # Should work with Tuple[int, int]
    _ = get_adaptive_pool(option="avg", output_size=(1, 1))

    # Should break if the tuple contains any non-ints
    with pytest.raises(TypeError):
        _ = get_adaptive_pool(option="avg", output_size=(1.0, 1))
    with pytest.raises(TypeError):
        _ = get_adaptive_pool(option="avg", output_size=(1, 1.0))
    with pytest.raises(TypeError):
        _ = get_adaptive_pool(option="avg", output_size=(1.0, 1.0))


def test_get_adaptive_pool_output_size_arg_length():
    """Test the length of the `output_size` arg of `get_adaptive_pool`.

    `output_size` should be a tuple of length 2.

    """
    # Should work with Tuple[int, int]
    _ = get_adaptive_pool(option="avg", output_size=(1, 1))

    # Should break if the tuple is not of length 2
    with pytest.raises(RuntimeError):
        _ = get_adaptive_pool(option="avg", output_size=(1,))
    with pytest.raises(RuntimeError):
        _ = get_adaptive_pool(option="avg", output_size=(1, 1, 1))


def test_get_adaptive_pool_output_size_arg_values():
    """Test values of elements in `output_size` arg of `get_adaptive_pool`.

    None of the values in output_size should be less than one.

    """
    # Should work with Tuple[int, int]
    _ = get_adaptive_pool(option="avg", output_size=(1, 1))

    with pytest.raises(ValueError):
        _ = get_adaptive_pool("avg", (1, 0))
    with pytest.raises(ValueError):
        _ = get_adaptive_pool("avg", (0, 1))
    with pytest.raises(ValueError):
        _ = get_adaptive_pool("avg", (0, 0))
