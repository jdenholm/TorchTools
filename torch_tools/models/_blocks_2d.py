"""Two-dimensional convolutional blocks."""
from torch import Tensor
from torch.nn import Module, Conv2d, BatchNorm2d, LeakyReLU

from torch_tools.models._argument_processing import (
    process_num_feats,
    process_negative_slope_arg,
)


class SingleConvBlock(Module):
    """Single 2D convolutional block.

    Parameters
    ----------
    in_chans : int
        The number of input channels the block should take.
    out_chans : int
        The number of output channels the block should produce.
    lr_slope : float,, optional
        The negative slope to use in the `LeakyReLU`.

    """

    def __init__(self, in_chans: int, out_chans: int, lr_slope: float = 0.1):
        """Build `SingleConvBlock`."""
        super().__init__()
        self.conv = Conv2d(
            process_num_feats(in_chans),
            process_num_feats(out_chans),
            kernel_size=3,
            padding=1,
        )
        self.batch_norm = BatchNorm2d(process_num_feats(out_chans))
        self.leaky_relu = LeakyReLU(negative_slope=process_negative_slope_arg(lr_slope))

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
        logits = self.conv(batch)
        normalised = self.batch_norm(logits)
        return self.leaky_relu(normalised)


class DoubleConvBlock(Module):
    """Double convolutional block."""

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
            lr_slope=process_negative_slope_arg(lr_slope),
        )
        self.conv2 = SingleConvBlock(
            process_num_feats(out_chans),
            process_num_feats(out_chans),
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
