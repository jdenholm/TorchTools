"""Miscellaneous helpful functions."""

from torch import Tensor


def batch_spatial_dims_divisible_by_2(batch: Tensor):
    """Check height and width of `batch` can be divided by 2.

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
        If the batch's images' heights are not divisible by 2.
    RuntimeError
        If the batch's images' heights are not divisible by 2.

    """
    if not isinstance(batch, Tensor):
        raise TypeError(f"'batch' should be a 'Tensor'. Got '{type(batch)}'.")
    if not batch.dim() == 4:
        msg = f"Mini-batch of images should have 4 dims. Got '{batch.dim()}'."
        raise RuntimeError(msg)

    _, _, height, width = batch.shape

    if (height % 2) != 0:
        msg = "Mini-batch of image-like's height should divide by 2. Got "
        msg += f"'{height}'."
        raise RuntimeError(msg)
    if (width % 2) != 0:
        msg = "Mini-batch of image-like's width should divide by 2. Got "
        msg += f"{width}."
        raise RuntimeError(msg)
