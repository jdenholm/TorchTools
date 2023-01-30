"""Tests for functions in `torch_tools.misc`."""

import pytest

from torch import rand  # pylint:disable=no-name-in-module

from torch_tools.misc import img_batch_dims_power_of_2, divides_by_two_check


def test_img_batch_dims_divisible_by_2_types():
    """Test the types accepted by the `batch` argument."""
    # Should work with 4D `Tensor` with height and width powers of 2
    img_batch_dims_power_of_2(rand(16, 3, 64, 64))

    # Should break with non-Tensor
    with pytest.raises(TypeError):
        img_batch_dims_power_of_2(1)
    with pytest.raises(TypeError):
        img_batch_dims_power_of_2([1, 2, 3])


def test_img_batch_dims_divisible_by_2_tensor_num_dimensions():
    """Test the number of dimensions accepted by `batch` argument."""
    # Should work with 4D `Tensor` with height and width powers of 2
    img_batch_dims_power_of_2(rand(16, 3, 64, 64))

    # Should break with non-4D `Tensor`.

    # With 1 dimension
    with pytest.raises(RuntimeError):
        img_batch_dims_power_of_2(rand(10))

    # With 2 dimensions
    with pytest.raises(RuntimeError):
        img_batch_dims_power_of_2(rand(10, 10))

    # With 3 dimensions
    with pytest.raises(RuntimeError):
        img_batch_dims_power_of_2(rand(10, 10, 10))

    # With 5 dimensions
    with pytest.raises(RuntimeError):
        img_batch_dims_power_of_2(rand(10, 10, 10, 10, 10))


def test_img_batch_spatial_dims_power_of_2_height_values():
    """Test the values accepted by the height dimension of `batch`."""
    # Should work with 4D `Tensor` with height and width powers of 2
    img_batch_dims_power_of_2(rand(16, 3, 64, 64))

    # Should break when the height dimension does not divide by 2
    with pytest.raises(RuntimeError):
        img_batch_dims_power_of_2(rand(16, 4, 6, 64))

    with pytest.raises(RuntimeError):
        img_batch_dims_power_of_2(rand(16, 4, 7, 64))

    with pytest.raises(RuntimeError):
        img_batch_dims_power_of_2(rand(16, 4, 9, 64))


def test_batch_dims_power_of_2_width_values():
    """Test the values accepted by the height dimension of `batch`."""
    # Should work with 4D `Tensor` with height and width powers of 2
    img_batch_dims_power_of_2(rand(16, 3, 64, 64))

    # Should break when the width dimension does not divide by 2
    with pytest.raises(RuntimeError):
        img_batch_dims_power_of_2(rand(16, 4, 64, 6))

    with pytest.raises(RuntimeError):
        img_batch_dims_power_of_2(rand(1, 4, 64, 7))

    with pytest.raises(RuntimeError):
        img_batch_dims_power_of_2(rand(16, 4, 64, 9))


def test_divides_by_two_check_types():
    """Test the types accepted by the `to_divide` argument."""
    # Should work with ints which can be divided by 2
    divides_by_two_check(10)

    # Should break with non-int
    with pytest.raises(TypeError):
        divides_by_two_check(2.0)
    with pytest.raises(TypeError):
        divides_by_two_check(2j)


def test_divides_by_two_check_values():
    """Test the values accepted by the `to_divide` argument."""
    # Should work with ints which can be divided by 2
    divides_by_two_check(10)

    # Should break with ints of zero or less
    with pytest.raises(ValueError):
        divides_by_two_check(0)
    with pytest.raises(ValueError):
        divides_by_two_check(-1)

    # Should break with positive ints which don't divide by 2
    with pytest.raises(ValueError):
        divides_by_two_check(1)
    with pytest.raises(ValueError):
        divides_by_two_check(3)
    with pytest.raises(ValueError):
        divides_by_two_check(5)
