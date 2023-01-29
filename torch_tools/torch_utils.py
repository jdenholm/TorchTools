"""PyTorch utilities."""

from torch import Tensor, eye  # pylint: disable=no-name-in-module


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
        Target Tensor of shape ``(num_classes, H, W)``.

    Raises
    ------
    TypeError
        If ``mask_img`` is not a ``Tensor``.
    TypeError
        If ``num_classes`` is not an ``int``.
    ValueError
        If any of the values in ``mask_img`` cannot be cast as int.
    ValueError
        If ``num_classes`` is less than two.
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
