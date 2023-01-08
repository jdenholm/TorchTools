"""Miscellaneous helpful functions."""

from torch import Tensor, log2, as_tensor  # pylint: disable=no-name-in-module


def batch_spatial_dims_power_of_2(batch: Tensor):
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
        msg += f"{width}."
        raise RuntimeError(msg)
