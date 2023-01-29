"""Tests for functions in ``torch_tools.torch_utils``."""
import pytest

from torch import randint, rand, full  # pylint: disable=no-name-in-module


from torch_tools.torch_utils import target_from_mask_img


def test_target_from_mask_img_is_tensor():
    """Test ``mask_img`` argument only accepts ``Tensor``."""
    # Should work with Tensor on [0, num_classes)
    _ = target_from_mask_img(randint(2, (8, 8)), num_classes=2)

    # Should break with non-Tensor
    with pytest.raises(TypeError):
        _ = target_from_mask_img(
            randint(2, (8, 8)).numpy(),
            num_classes=2,
        )

    with pytest.raises(TypeError):
        _ = target_from_mask_img([1, 2, 3], num_classes=2)


def test_target_from_mask_img_num_classes_is_int():
    """Test ``num_classes`` only accepts ints."""
    # Should work with positive int
    _ = target_from_mask_img(randint(2, (8, 8)), num_classes=2)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = target_from_mask_img(randint(2, (8, 8)), num_classes=2.0)


def test_target_from_mask_img_values_cast_as_int():
    """Test ``mask_img`` only works with integer-like."""
    # Should work with positive integer-like
    _ = target_from_mask_img(randint(2, (8, 8)), num_classes=2)
    _ = target_from_mask_img(randint(2, (8, 8)).float(), num_classes=2)

    # Should break with non-integer-like
    # Non-integer-like are numbers where (x % 1) != 0
    with pytest.raises(ValueError):
        _ = target_from_mask_img(rand(8, 8), num_classes=2)


def test_target_from_mask_img_values_are_at_least_two():
    """Make sure ``num_classes`` arg is at least two."""
    # Should work with ints of two or more
    _ = target_from_mask_img(randint(2, (8, 8)), num_classes=2)
    _ = target_from_mask_img(randint(2, (8, 8)), num_classes=3)

    # Should break with ints less than 2
    with pytest.raises(ValueError):
        _ = target_from_mask_img(randint(2, (8, 8)), num_classes=1)
    with pytest.raises(ValueError):
        _ = target_from_mask_img(randint(2, (8, 8)), num_classes=0)
    with pytest.raises(ValueError):
        _ = target_from_mask_img(randint(2, (8, 8)), num_classes=-1)


def test_target_from_mask_img_value_range():
    """Make sure the values in ``mask_img`` are on ``[0, num_classes)``."""
    # Should work with values on [0, num_classes)
    _ = target_from_mask_img(randint(5, (8, 8)), num_classes=5)

    # Should break with values less than zero
    with pytest.raises(ValueError):
        _ = target_from_mask_img(full((8, 8), -1), num_classes=2)

    # Should break with values equal to num_classes
    with pytest.raises(ValueError):
        _ = target_from_mask_img(full((8, 8), 10), num_classes=10)

    # Should break with values greater than num_classes
    with pytest.raises(ValueError):
        _ = target_from_mask_img(full((8, 8), 21), num_classes=20)


def test_target_from_mask_img_dimensions():
    """Make sure ``mask_img`` is two-dimensional."""
    # Should work with 2D inputs
    _ = target_from_mask_img(randint(5, (8, 8)), num_classes=5)

    # Should break with non-2d inputs
    with pytest.raises(RuntimeError):
        _ = target_from_mask_img(randint(5, (8,)), num_classes=5)
    with pytest.raises(RuntimeError):
        _ = target_from_mask_img(randint(5, (8, 8, 8)), num_classes=5)
    with pytest.raises(RuntimeError):
        _ = target_from_mask_img(randint(5, (8, 8, 8, 8)), num_classes=5)


def test_target_from_mask_img_return_values():
    """Test the value returned are correct."""
    for _ in range(5):
        mask_img = randint(100, (64, 64))
        target = target_from_mask_img(mask_img, num_classes=100)
        assert (target.argmax(dim=0) == mask_img).all()
