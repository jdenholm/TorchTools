"""Tests for functions in `torch_tools.misc`."""

import pytest

from torch import rand  # pylint:disable=no-name-in-module

from torch_tools.misc import batch_spatial_dims_power_of_2


def test_batch_spatial_dims_divisible_by_2_types():
    """Test the types accepted by the `batch` argument."""
    # Should work with 4D `Tensor` with height and width powers of 2
    batch_spatial_dims_power_of_2(rand(16, 3, 64, 64))

    # Should break with non-Tensor
    with pytest.raises(TypeError):
        batch_spatial_dims_power_of_2(1)
    with pytest.raises(TypeError):
        batch_spatial_dims_power_of_2([1, 2, 3])


def test_batch_spatial_dims_divisible_by_2_tensor_num_dimensions():
    """Test the number of dimensions accepted by `batch` argument."""
    # Should work with 4D `Tensor` with height and width powers of 2
    batch_spatial_dims_power_of_2(rand(16, 3, 64, 64))

    # Should break with non-4D `Tensor`.

    # With 1 dimension
    with pytest.raises(RuntimeError):
        batch_spatial_dims_power_of_2(rand(10))

    # With 2 dimensions
    with pytest.raises(RuntimeError):
        batch_spatial_dims_power_of_2(rand(10, 10))

    # With 3 dimensions
    with pytest.raises(RuntimeError):
        batch_spatial_dims_power_of_2(rand(10, 10, 10))

    # With 5 dimensions
    with pytest.raises(RuntimeError):
        batch_spatial_dims_power_of_2(rand(10, 10, 10, 10, 10))


def test_batch_spatial_dims_power_of_2_height_values():
    """Test the values accepted by the height dimension of `batch`."""
    # Should work with 4D `Tensor` with height and width powers of 2
    batch_spatial_dims_power_of_2(rand(16, 3, 64, 64))

    # Should break when the height dimension does not divide by 2
    with pytest.raises(RuntimeError):
        batch_spatial_dims_power_of_2(rand(16, 4, 6, 64))

    with pytest.raises(RuntimeError):
        batch_spatial_dims_power_of_2(rand(16, 4, 7, 64))

    with pytest.raises(RuntimeError):
        batch_spatial_dims_power_of_2(rand(16, 4, 9, 64))


def test_batch_spatial_dims_power_of_2_width_values():
    """Test the values accepted by the height dimension of `batch`."""
    # Should work with 4D `Tensor` with height and width powers of 2
    batch_spatial_dims_power_of_2(rand(16, 3, 64, 64))

    # Should break when the width dimension does not divide by 2
    with pytest.raises(RuntimeError):
        batch_spatial_dims_power_of_2(rand(16, 4, 64, 6))

    with pytest.raises(RuntimeError):
        batch_spatial_dims_power_of_2(rand(1, 4, 64, 7))

    with pytest.raises(RuntimeError):
        batch_spatial_dims_power_of_2(rand(16, 4, 64, 9))
