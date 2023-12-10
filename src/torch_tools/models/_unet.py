"""UNet model for semantic segmentation."""
from typing import List
from torch import Tensor

from torch.nn import Module, Conv2d, ModuleList

from torch_tools.models._argument_processing import (
    process_num_feats,
    process_u_architecture_layers,
    process_2d_kernel_size,
    process_negative_slope_arg,
    process_2d_block_style_arg,
    process_boolean_arg,
)

from torch_tools.models._blocks_2d import (
    DoubleConvBlock,
    DownBlock,
    UNetUpBlock,
    ConvResBlock,
)


# pylint: disable=too-many-arguments


class UNet(Module):
    """UNet model for semantic segmentation.

    Parameters
    ----------
    in_chans : int
        The number of input channels.
    out_chans : int
        The number of output channels.
    features_start : int, optional
        The number of features produced by the first convolutional block.
    num_layers : int, optional
        The number of layers in the ``UNet``.
    pool_style : str, optional
        The pool style to use in the ``DownBlock`` blocks. Can be ``"max"`` or
        ``"avg"``.
    bilinear : bool, optional
        Whether to use use bilinear interpolation in the upsampling layers or
        not. If ``True``, we use bilinear interpolation to upsample. If
        ``False``, we use ``ConvTranspose2d``.
    lr_slope : float, optional
        The negative slope argument for ``LeakyReLU`` layers.
    kernel_size : int, optional
        Linear size of the square convolutional kernel to use in the ``Conv2d``
        layers. Should be a positive, odd, int.
    block_style : str
        Type of convolutional blocks to use: ``"double_conv"`` or
        ``"conv_res"``.


    Examples
    --------
    >>> from torch_tools import UNet
    >>> model = UNet(
                    in_chans=3,
                    out_chans=16,
                    features_start=64,
                    num_layers=3,
                    pool_style="max",
                    bilinear=False,
                    lr_slope=0.2,
                    kernel_size=3,
                    )

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
        kernel_size: int = 3,
        block_style: str = "double_conv",
    ):
        """Build `UNet`."""
        super().__init__()

        self.in_conv = self._get_input_block(
            process_num_feats(in_chans),
            process_num_feats(features_start),
            process_negative_slope_arg(lr_slope),
            process_2d_kernel_size(kernel_size),
            process_2d_block_style_arg(block_style),
        )

        self.down_blocks = self._get_down_blocks(
            process_u_architecture_layers(num_layers),
            process_num_feats(features_start),
            pool_style,
            process_negative_slope_arg(lr_slope),
            process_2d_kernel_size(kernel_size),
            process_2d_block_style_arg(block_style),
        )

        self.up_blocks = self._get_up_blocks(
            process_u_architecture_layers(num_layers),
            process_num_feats(features_start),
            process_boolean_arg(bilinear),
            process_negative_slope_arg(lr_slope),
            process_2d_kernel_size(kernel_size),
            process_2d_block_style_arg(block_style),
        )

        self.out_conv = Conv2d(
            process_num_feats(features_start),
            process_num_feats(out_chans),
            kernel_size=1,
            stride=1,
        )

    def _get_input_block(
        self,
        in_chans: int,
        features_start: int,
        lr_slope: float,
        kernel_size: int,
        block_style: str,
    ) -> Module:
        """Return the UNet's input block.

        Parameters
        ----------
        in_chans : int
            The number input channels to the model.
        features_start : int
            The number of features the input block should produce.
        lr_slope : float
            The negative slope argument in the leaky relu layers.
        kernel_size : int
            Length of the square convolutional kernel.
        block_style : str
            The style of the block.

        Returns
        -------
        block : Module
            The input convolutional block.

        """
        if block_style == "double_conv":
            block: Module = DoubleConvBlock(
                in_chans=in_chans,
                out_chans=features_start,
                lr_slope=lr_slope,
                kernel_size=kernel_size,
            )
        else:
            block = ConvResBlock(
                in_chans=in_chans,
                out_chans=features_start,
                lr_slope=lr_slope,
                kernel_size=kernel_size,
            )

        return block

    def _get_down_blocks(
        self,
        num_layers: int,
        features_start,
        pool_style: str,
        lr_slope: float,
        kernel_size: int,
        block_style: str,
    ) -> ModuleList:
        """Stack the downsampling blocks in a `ModuleList`.

        Parameters
        ----------
        num_layers : int
            The number of user-requested layers in the U.
        features_start : int
            The number of features produced by the input conv block.
        pool_style : str
            The style of pool to use in the ``DownBlock`` blocks.
        lr_slope : float
            The negative slope are for ``DownBlock`` (negative slope in
            ``LeakyReLU`` layers.)
        kernel_size : int, optional
            Linear size of the square convolutional kernel to use in the
            ``Conv2d`` layers. Should be a positive, odd, int.
        block_style : str
            Type of convolutional block to use.

        Returns
        -------
        ModuleList
            A ``ModuleList`` holding the downsampling blocks.

        """
        chans = features_start
        blocks = []
        for _ in range(num_layers - 1):
            blocks.append(
                DownBlock(
                    chans,
                    chans * 2,
                    pool_style,
                    lr_slope,
                    kernel_size=kernel_size,
                    block_style=block_style,
                )
            )
            chans *= 2

        return ModuleList(blocks)

    def _get_up_blocks(
        self,
        num_layers: int,
        features_start: int,
        bilinear: bool,
        lr_slope: float,
        kernel_size: int,
        block_style: str,
    ) -> ModuleList:
        """Stack the upsampling blocks in a ``ModuleList``.

        Parameters
        ----------
        num_layers : int
            The number of layers requested in the U.
        features_start : int
            The number of features produced by the first conv block.
        bilinear : bool
            Whether the upsamplping should be done with bilinear interpolation
            or conv transpose.
        lr_slope : float
            The negative slope to use in the ``LeakReLU``s.
        kernel_size : int, optional
            Linear size of the square convolutional kernel to use in the
            ``Conv2d`` layers. Should be a positive, odd, int.
        block_style : str
            Style of convolutional blocks to use.

        Returns
        -------
        ModuleList
            The upsampling layers stacked in a ``ModuleList``.

        """
        chans = features_start * (2 ** (num_layers - 1))
        blocks = []
        for _ in range(num_layers - 1):
            blocks.append(
                UNetUpBlock(
                    chans,
                    chans // 2,
                    bilinear,
                    lr_slope,
                    kernel_size=kernel_size,
                    block_style=block_style,
                )
            )
            chans //= 2
        return ModuleList(blocks)

    def _down_forward_pass(self, batch: Tensor) -> List[Tensor]:
        """Perform the UNet's downsampling forward pass..

        Parameters
        ----------
        batch : Tensor
            A min-batch input.

        Returns
        -------
        down_features : List[Tensor]
            A list of the features produced by each downsampling layer,
            with ``batch`` at element zero.

        """
        down_features = [batch]
        for down_layer in self.down_blocks:
            down_features.append(down_layer(down_features[-1]))
        return down_features

    def _up_forward_pass(self, down_features: List[Tensor]) -> Tensor:
        """Perform the UNet's upsampling forward pass.

        Parameters
        ----------
        down_features : List[Tensor]
            List of the down half of the UNet's features (and input).

        Returns
        -------
        up_batch : Tensor
            The up-sampled batch.

        """
        up_batch = self.up_blocks[0](down_features[-1], down_features[-2])
        # Iterate over the remaining up layers zipped with the
        # third-last to zeroth down features.
        for up_conv, feat in zip(self.up_blocks[1:], down_features[::-1][2:]):
            up_batch = up_conv(up_batch, feat)
        return up_batch

    def forward(self, batch: Tensor) -> Tensor:
        """Pass ``batch`` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of image-like inputs.

        Returns
        -------
        Tensor
            The result of passing ``batch`` through the model.

        """
        batch = self.in_conv(batch)
        down_features = self._down_forward_pass(batch)
        return self.out_conv(self._up_forward_pass(down_features))
