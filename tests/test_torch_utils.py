"""Tests for functions in ``torch_tools.torch_utils``."""
import pytest

from torch import randint, rand, full, zeros, ones  # pylint: disable=no-name-in-module


from torch_tools.torch_utils import (
    target_from_mask_img,
    patchify_img_batch,
    img_batch_dims_power_of_2,
)

# pylint: disable=redefined-outer-name


@pytest.fixture()
def create_fake_image_batch():
    """Create a fake batch of images to test with."""
    batch = zeros(10, 3, 64, 128)

    counter = 0
    for idx in range(10):
        for row in range(0, 64, 4):
            for col in range(0, 128, 4):
                for chan in range(3):
                    value = counter + (chan * 0.1)
                    batch[idx, chan, row : (row + 4), col : (col + 4)] = value
                counter += 1
    return batch


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


def test_patchify_img_batch_img_batch_arg_type():
    """Test the ``img_batch`` argument only accepts ``Tensor``."""
    # Should work with mini-batches of img-like
    patchify_img_batch(rand(1, 3, 50, 50), patch_size=10)

    # Should break with non-Tensor
    with pytest.raises(TypeError):
        patchify_img_batch(rand(1, 3, 50, 50).numpy(), patch_size=10)


def test_patchify_img_batch_img_batch_dimensions():
    """Test the dimensions accepted by ``img_batch``."""
    # Should work with four-dimensional image batches
    patchify_img_batch(rand(1, 3, 50, 50), patch_size=10)

    # Should break with non-4D inputs
    with pytest.raises(RuntimeError):
        patchify_img_batch(rand(10), patch_size=10)
    with pytest.raises(RuntimeError):
        patchify_img_batch(rand(6, 6), patch_size=2)
    with pytest.raises(RuntimeError):
        patchify_img_batch(rand(6, 6, 6), patch_size=2)
    with pytest.raises(RuntimeError):
        patchify_img_batch(rand(2, 2, 2, 2, 2), patch_size=2)


def test_patchify_img_batch_patch_size_arg_type():
    """Test the ``patch_size`` argument only accepts ``int``."""
    # Should work with int
    patchify_img_batch(rand(1, 3, 50, 50), patch_size=10)

    # Should break with non-int
    with pytest.raises(TypeError):
        patchify_img_batch(rand(1, 3, 50, 50), patch_size=10.0)
    with pytest.raises(TypeError):
        patchify_img_batch(rand(1, 3, 50, 50), patch_size=1j)


def test_patchify_img_batch_zero_or_negative_patch_size():
    """Test zero or negative ``patch_size``s are caught."""
    # Should work with positive int
    patchify_img_batch(rand(1, 3, 50, 50), patch_size=10)

    with pytest.raises(ValueError):
        patchify_img_batch(rand(1, 3, 50, 50), patch_size=0)
    with pytest.raises(ValueError):
        patchify_img_batch(rand(1, 3, 50, 50), patch_size=-1)
    with pytest.raises(ValueError):
        patchify_img_batch(rand(1, 3, 50, 50), patch_size=-2)


def test_patchify_img_batch_patch_and_img_size_mismatch():
    """Test the ``patch_size`` divides the image height and width."""
    # Should work if the patch_size divides the image height and width
    patchify_img_batch(rand(1, 3, 40, 20), patch_size=10)
    patchify_img_batch(rand(1, 3, 20, 40), patch_size=4)
    patchify_img_batch(rand(1, 3, 40, 40), patch_size=2)

    # Should break if the patch size doesn't divide height
    with pytest.raises(ValueError):
        patchify_img_batch(rand(1, 3, 41, 40), patch_size=2)
    with pytest.raises(ValueError):
        patchify_img_batch(rand(1, 3, 40, 41), patch_size=2)


def test_patchify_img_batch_return_size():
    """Test the dimensionality of the returned batches."""
    batch = rand(1, 3, 512, 512)
    patched = patchify_img_batch(batch, patch_size=64)
    assert patched.shape == (64, 3, 64, 64)

    batch = rand(10, 3, 512, 512)
    patched = patchify_img_batch(batch, patch_size=64)
    assert patched.shape == (640, 3, 64, 64)


def test_patchify_img_batch_return_values(create_fake_image_batch):
    """Test the values returned in each patch.

    Notes
    -----
    See ``create_fake_image_batch``.

    """
    batch = create_fake_image_batch

    patches = patchify_img_batch(batch, 4)
    for idx, patch in enumerate(patches):
        assert (patch[0, :, :] == idx).all()
        assert (patch[1, :, :] == idx + 0.1).all()
        assert (patch[2, :, :] == idx + 0.2).all()


def test_patchify_img_batch_allows_gradient_flow():
    """Test gradient flow works through function."""
    batch = ones(10, 3, 32, 32, requires_grad=True)
    patches = patchify_img_batch(batch, 4)
    out = (patches * 2).sum()
    out.backward()

    assert (batch.grad == 2).all()

    batch = ones(10, 3, 32, 32, requires_grad=True)
    patches = patchify_img_batch(batch, 4)
    out = (patches * 123).sum()
    out.backward()

    assert (batch.grad == 123).all()
