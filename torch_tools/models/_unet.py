"""UNet model for semantic segmentation."""

from torch import Tensor

from torch.nn import Module, Conv2d

from torch_tools.models._argument_processing import (
    process_num_feats,
    process_negative_slope_arg,
)

from torch_tools.models._blocks_2d import DoubleConvBlock


# pylint : disable=too-many-arguments


class UNet(Module):
    """UNet model for semantic segmentation.

    Parameters
    ----------
    in_chans : int
        The number of input channels.
    out_chans : int
        The number of output channels.
    features_start : int
        The number of features produced by the first convolutional block.
    num_layers : int
        The number of layers in the `UNet`.
    pool_style : str
        The pool style to use in the `DownBlock`s. Can be `"max"` or `"avg"`.
    bilinear : bool
        Whether to use use bilinear interpolation in the upsampling layers or
        not. If `True`, we use bilinear interpolation to upsample. If `False`,
        we use `ConvTranspose2d`.
    lr_slope : float
        The negative slope argument for `LeakyReLU` layers.

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        features_start: int = 64,
        num_layers: int = 4,
        pool_style: str = "max",
        bilinear: bool = False,
        lr_slope: float = 0.1,
    ):
        """Build `UNet`."""
        super().__init__()

        self._in_conv = DoubleConvBlock(
            in_chans=process_num_feats(in_chans),
            out_chans=process_num_feats(features_start),
            lr_slope=process_negative_slope_arg(lr_slope),
        )

        self._out_conv = Conv2d(
            process_num_feats(features_start),
            process_num_feats(out_chans),
            kernel_size=1,
            stride=1,
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
            The resul of passing batch through the model.

        """
        raise NotImplementedError("UNet forward method not implemented.")
