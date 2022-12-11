"""Two-dimensional convolutional blocks."""
from typing import List

from torch import Tensor
from torch.nn import Module, Conv2d, BatchNorm2d, LeakyReLU, Sequential, ReLU

from torch_tools.models._argument_processing import (
    process_num_feats,
    process_negative_slope_arg,
    process_boolean_arg,
)


class SingleConvBlock(Module):
    """Single 2D convolutional block.

    Parameters
    ----------
    in_chans : int
        The number of input channels the block should take.
    out_chans : int
        The number of output channels the block should produce.
    batch_norm : bool
        Should we include a `BatchNorm2d` layer?
    leaky_relu : bool
        Should we include a `LeakyReLU` layer?
    lr_slope : float,, optional
        The negative slope to use in the `LeakyReLU` (use 0.0 for `ReLU`).

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        batch_norm: bool = True,
        leaky_relu: bool = True,
        lr_slope: float = 0.1,
    ):
        """Build `SingleConvBlock`."""
        super().__init__()
        self._fwd = self._layers(
            process_num_feats(in_chans),
            process_num_feats(out_chans),
            process_boolean_arg(batch_norm),
            process_boolean_arg(leaky_relu),
            process_negative_slope_arg(lr_slope),
        )

    @staticmethod
    def _layers(
        in_chans: int,
        out_chans: int,
        batch_norm: bool,
        leaky_relu: bool,
        lr_slope: float,
    ) -> Sequential:
        """Stack the block's layers in a `Sequential`.

        Parameters
        ----------
        (See class docstring).

        Returns
        -------
        Sequential
            The block's layers in a `Sequential`.

        """
        layers: List[Module]
        layers = [Conv2d(in_chans, out_chans, kernel_size=3, padding=1)]

        if batch_norm is True:
            layers.append(BatchNorm2d(out_chans))

        if leaky_relu is True:
            layers.append(LeakyReLU(lr_slope))

        return Sequential(*layers)

    def forward(self, batch: Tensor) -> Tensor:
        """Pass `batch` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        Returns
        -------
        Tensor
            The result of passing `batch` through the model.

        """
        return self._fwd(batch)


class DoubleConvBlock(Module):
    """Double convolutional block.

    Parameters
    ----------
    in_chans : int
        The number of input channels the block should take.
    out_chans : int
        The number of output channels the block should produce.
    lr_slope : float, optional
        The slope to use in the `LeakyReLU` layers.

    """

    def __init__(self, in_chans: int, out_chans: int, lr_slope: float = 0.1):
        """Build `DoubleConvBlock`.

        Parameters
        ----------
        in_chans : int
            The number of input channels the block should take.
        out_chans : int
            The number of output channels the block should take.
        lr_slope : float, optional
            The negative slope to use in the `LeakyReLU`.

        """
        super().__init__()
        self.conv1 = SingleConvBlock(
            process_num_feats(in_chans),
            process_num_feats(out_chans),
            batch_norm=True,
            leaky_relu=True,
            lr_slope=process_negative_slope_arg(lr_slope),
        )
        self.conv2 = SingleConvBlock(
            process_num_feats(out_chans),
            process_num_feats(out_chans),
            batch_norm=True,
            leaky_relu=True,
            lr_slope=process_negative_slope_arg(lr_slope),
        )

    def forward(self, batch: Tensor) -> Tensor:
        """Pass `batch` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        Returns
        -------
        Tensor
            The result of passing `batch` through the model.

        """
        return self.conv2(self.conv1(batch))


class ResidualBlock(Module):
    """Residual block."""

    def __init__(self, num_chans: int):
        """Build `ResidualBlock`."""
        super().__init__()
        self.conv1 = SingleConvBlock(
            num_chans,
            num_chans,
            batch_norm=True,
            leaky_relu=True,
            lr_slope=0.0,
        )
        self.conv2 = SingleConvBlock(
            num_chans,
            num_chans,
            batch_norm=True,
            leaky_relu=False,
        )

        self.relu = ReLU()

    def forward(self, batch: Tensor) -> Tensor:
        """Pass `batch` through the block.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        Returns
        -------
        Tensor
            The result of passing `batch` through the block.

        """
        identity = batch
        out = self.conv1(batch)
        out = self.conv2(out)
        out += identity
        return self.relu(out)
