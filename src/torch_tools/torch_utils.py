"""PyTorch utilities."""
from itertools import chain

from torch import (  # pylint: disable=no-name-in-module
    Tensor,
    eye,
    concat,
    log2,
    as_tensor,
    square,
    sqrt,
    abs,
)


from torch.nn import Module


def target_from_mask_img(mask_img: Tensor, num_classes: int) -> Tensor:
    """Convert 1-channel image to a target tensor for semantic segmentation.

    Parameters
    ----------
    mask_img : Tensor
        An image holding the segmentation mask. Should be on [0, num_classes)
        with shape ``(H, W)``, where ``H`` is the image height and ``W`` the
        width.


    Returns
    -------
    Tensor
        Target Tensor of shape ``(num_classes, H, W)``. Each element,
        ``target[:, i, j]`` is a one-hot-encoded vector.

    Raises
    ------
    TypeError
        If ``mask_img`` is not a ``Tensor``.
    TypeError
        If ``num_classes`` is not an ``int``.
    ValueError
        If any of the values in ``mask_img`` cannot be cast as int.
    ValueError
        If ``num_classes < 2``.
    ValueError
        If ``mask_img`` has values less than zero, or greater than/equal to
        ``num_classes``.
    RuntimeError
        If ``mask_img`` is not two-dimensional.

    """
    if not isinstance(mask_img, Tensor):
        msg = f"'mask_img' should be a Tensor. Got {type(mask_img)}."
        raise TypeError(msg)
    if not isinstance(num_classes, int):
        msg = f"'num_classes' should be an int. Got {type(num_classes)}."
        raise TypeError(msg)
    if not (mask_img % 1 == 0).all():
        msg = "'mask_img' values should have no remainder when dividing by "
        msg += f"1. Got values '{mask_img.unique()}'."
        raise ValueError(msg)
    if num_classes < 2:
        msg = "There should be a minimum of two classes (foreground and "
        msg += f"background). Got '{num_classes}'."
        raise ValueError(msg)
    if mask_img.min() < 0 or mask_img.max() >= num_classes:
        msg = f"'mask_img' values should be on [0, {num_classes}). Got "
        msg += f"values on [{mask_img.min()}, {mask_img.max()}]."
        raise ValueError(msg)
    if mask_img.dim() != 2:
        msg = f"'mask_img' should have two dimensions. Got {mask_img.dim()}."
        raise RuntimeError(msg)
    return eye(num_classes)[mask_img.long()].permute(2, 0, 1)


def _img_batch_check(img_batch: Tensor):
    """Run checks on ``img_batch``.

    Parameters
    ----------
    img_batch : Tensor
        A mini-batch of image-like.

    Raises
    ------
    TypeError
        If mini-batch is not a ``Tensor``.
    RuntimeError
        If ``img_batch`` is not four-dimensional.

    """
    if not isinstance(img_batch, Tensor):
        msg = f"'img_batch' should be Tensor. Got '{type(img_batch)}'."
        raise TypeError(msg)
    if not img_batch.dim() == 4:
        msg = f"'img_batch' should be 4D. Got '{img_batch.dim()}' dimensions."
        raise RuntimeError(msg)


def _patch_size_check(img_batch: Tensor, patch_size: int):
    """Run checks on ``patch_size``.

    Parameters
    ----------
    img_batch : Tensor
        A mini-batch of images.
    patch_size : int
        Size of the square mini-patches to split the images into.

    Raises
    ------
    TypeError
        If ``patch_size`` is not an int.
    ValueError
        If ``patch_size <= 0``.
    ValueError
        If ``patch_size`` does evenly divide the image height or width.

    """
    if not isinstance(patch_size, int):
        msg = f"'patch_size' should be int. Got '{type(patch_size)}'."
        raise TypeError(msg)

    if patch_size <= 0:
        msg = f"'patch_size' should exceed zero. Got '{patch_size}'."
        raise ValueError(msg)

    _, _, height, width = img_batch.shape

    if (height % patch_size != 0) or (width % patch_size != 0):
        msg = f"'patch_size' '{patch_size}' should divide image height and "
        msg += f"width '{(height, width)}'."
        raise ValueError(msg)


def patchify_img_batch(img_batch: Tensor, patch_size: int) -> Tensor:
    """Turn ``img_batch`` into a collection of patches.

    Note: gradient flow works through this function.

    Parameters
    ----------
    img : Tensor
        Convert ``img_batch`` into a batch of sub-patches. Should have size
        ``(N, C, H, W)``, where ``N`` is the batch size, ``C`` is the number
        of channels, ``H`` is the image height and ``W`` the width.
    patch_size : int
        Size of the square patches to break the images in ``img_batch`` into.

    Returns
    -------
    Tensor
        ``img_batch`` as a collection of small patches. The returned
        ``Tensor`` has size
        ``(N * H / patch_size * W / patch_size, C, patch_size, patch_size)``.
        For example: using a batch of 10 RGB images of size 16x16, and a patch
        size of 4, will return a ``Tensor`` of shape ``(160, 3, 4, 4)``.

    """
    _img_batch_check(img_batch)
    _patch_size_check(img_batch, patch_size)

    _, channels, _, _ = img_batch.shape

    unfolded = (
        concat(list(img_batch), dim=1)
        .unfold(0, channels, channels)
        .unfold(1, patch_size, patch_size)
        .unfold(2, patch_size, patch_size)
    )

    unfolded_list = list(chain(*chain(*unfolded)))
    return concat(list(map(lambda x: x.unsqueeze(0), unfolded_list)), dim=0)


def img_batch_dims_power_of_2(batch: Tensor):
    """Check height and width of ``batch`` are powers of 2.

    Parameters
    ----------
    batch : Tensor
        A mini-batch of image-like inputs.

    Raises
    ------
    TypeError
        If ``batch`` is not a ``Tensor``.
    RuntimeError
        If ``batch`` does not have four dimensions.
    RuntimeError
        If the ``batch``'s images' heights are not a power of 2.
    RuntimeError
        If the ``batch``'s images' heights are not a power of 2.

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


def disable_biases(model: Module):
    """Disable all ``bias`` parameters in model.

    Parameters
    ----------
    model : Module
        The model to disable biases in.

    """
    if not isinstance(model, Module):
        msg = "'disable_biases' function should only be applied to "
        msg += f"'torch.nn.Module'. Got '{type(model)}'."
        raise TypeError(msg)

    if hasattr(model, "bias"):
        model.bias = None  # type: ignore


def total_image_variation(
    img_batch: Tensor,
    mean_reduce: bool = True,
) -> Tensor:
    """Compute the total variation of ``img_batch``.

    Parameters
    ----------
    img_batch : Tensor
        A mini-batch of image-like inputs.
    mean_reduce : bool, optional
        If ``True``, the returned value is normalised by the number of
        elements in ``img_batch``. If ``False``, no division is performed.

    Returns
    -------
    Tensor
        The total variation measure.

    Raises
    ------
    TypeError
        If ``img_batch`` is not a ``Tensor``.
    TypeError
        If ``mean_reduce`` is not a ``bool``.
    RuntimeError
        If ``img_batch`` is not a 4D ``Tensor``.

    Notes
    -----
    Heavily inspired by the TensorFlow code:
    https://github.com/tensorflow/tensorflow/blob/v2.14.0/tensorflow/python/ops/image_ops_impl.py#L3322-L3391

    """
    if not isinstance(img_batch, Tensor):
        msg = f"'img_batch' should be 'Tensor', got {type(img_batch)}."
        raise TypeError(msg)

    if not isinstance(mean_reduce, bool):
        msg = f"'mean_reduce' should be 'bool', got {type(mean_reduce)}."
        raise TypeError(msg)

    if not img_batch.ndim == 4:
        msg = f"'img_batch' should be 4D, but got {img_batch.ndim}D instead."
        raise RuntimeError(msg)

    row = abs(img_batch[:, :, 1:, :] - img_batch[:, :, :-1, :]).sum()
    col = abs(img_batch[:, :, :, 1:] - img_batch[:, :, :, :-1]).sum()

    total = (row + col).sum()

    return total / img_batch.numel() if mean_reduce is True else total
