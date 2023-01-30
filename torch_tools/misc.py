"""Miscellaneous helpful functions."""

from torch import Tensor, log2, as_tensor  # pylint: disable=no-name-in-module


def img_batch_dims_power_of_2(batch: Tensor):
    """Check height and width of `batch` are powers of 2.

    Parameters
    ----------
    batch : Tensor
        A mini-batch of image-like inputs.

    Raises
    ------
    TypeError
        If `batch` is not a `Tensor`.
    RuntimeError
        If `batch` does not have four dimensions.
    RuntimeError
        If the batch's images' heights are not a power of 2.
    RuntimeError
        If the batch's images' heights are not a power of 2.

    """
    if not isinstance(batch, Tensor):
        raise TypeError(f"'batch' should be a 'Tensor'. Got '{type(batch)}'.")
    if not batch.dim() == 4:
        msg = f"Mini-batch of images should have 4 dims. Got '{batch.dim()}'."
        raise RuntimeError(msg)

    _, _, height, width = batch.shape

    if (log2(as_tensor(height)) % 1) != 0:
        msg = "Mini-batch of image-like's height should be power of 2. Got "
        msg += f"'{height}'."
        raise RuntimeError(msg)
    if (log2(as_tensor(width)) % 1) != 0:
        msg = "Mini-batch of image-like's width should be power of 2. Got "
        msg += f"'{width}'."
        raise RuntimeError(msg)


def divides_by_two_check(to_divide: int):
    """Make sure `to_divide` can be divided by 2.

    Parameters
    ----------
    to_divide : int
        A number to be divided by two.

    Raises
    ------
    TypeError
        If `to_divide` is not an int.
    ValueError
        If `to_divide` is not greater than zero.
    ValueError
        If `to_divide / 2` is irrational.

    """
    if not isinstance(to_divide, int):
        raise TypeError(f"'to_divide' should be in. Got '{type(to_divide)}'.")
    if to_divide <= 0:
        msg = f"'to_divide' should be greater than zero. Got '{to_divide}'"
        raise ValueError(msg)
    if (to_divide % 2) != 0:
        raise ValueError(f"'{to_divide}' does not divide by 2.")
