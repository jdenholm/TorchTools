"""Utility functions for creating 2D adaptive pooling layers."""
from typing import Tuple

from torch import Tensor, concat
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d, Module

from torch_tools.models._argument_processing import (
    process_adaptive_pool_output_size_arg,
)


class _ConcatMaxAvgPool2d(Module):
    """Adaptive 2D pooling layer.

    Parameters
    ----------
    output_size : Tuple[int, int]
        Ouput size arg for adaptive pools.

    """

    def __init__(self, output_size: Tuple[int, int]):
        """Build `AdaptivePoool2d`."""
        super().__init__()
        self._avg_pool = AdaptiveAvgPool2d(output_size)
        self._max_pool = AdaptiveMaxPool2d(output_size, return_indices=False)

    def forward(self, batch: Tensor) -> Tensor:
        """Pass `batch` through the pooling layer.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        Returns
        -------
        Tensor
            The 'pooled' mini-batch.

        """
        return concat([self._avg_pool(batch), self._max_pool(batch)], dim=1)


_options = {
    "avg": AdaptiveAvgPool2d,
    "max": AdaptiveMaxPool2d,
    "avg-max-concat": _ConcatMaxAvgPool2d,
}


def get_adaptive_pool(option: str, output_size: Tuple[int, int]) -> Module:
    """Return the adaptive pooling layer.

    Parameters
    ----------
    option : str
        Adaptive pool option: 'avg', 'max', 'avg-max-concat'.
    output_size : Tuple[int, int]
        The output size the pooling layer should produce.

    Returns
    -------
    Module
        The adaptive pooling layer of choice.

    Raises
    ------
    TypeError
        If `option` is not a str.
    RuntimeError
        If `option` is not in `_options`.

    """
    if not isinstance(option, str):
        raise TypeError(f"Encoder option should be str. Got '{type(option)}'.")

    process_adaptive_pool_output_size_arg(output_size)

    if option not in _options:
        msg = f"Encoder option '{option}' no supported. Choose from "
        msg += f"{list(_options.keys())}."
        raise RuntimeError(msg)

    return _options[option](output_size=output_size)
