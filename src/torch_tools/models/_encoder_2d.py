"""Two-dimensional convolutional encoder moder."""
from typing import List, Optional

from torch.nn import Sequential


from torch_tools.models._blocks_2d import DownBlock, DoubleConvBlock
from torch_tools.models._argument_processing import (
    process_num_feats,
    process_str_arg,
    process_negative_slope_arg,
    process_u_architecture_layers,
    process_2d_kernel_size,
    process_max_feats_arg,
)

# pylint: disable=too-many-arguments


class Encoder2d(Sequential):
    """Encoder model for image-like inputs.

    A ``DoubleConvBlock`` which produces ``start_features`` features, followed
    by ``num_blocks - 1`` ``DownBlock`` blocks. The ``DoubleConvBlock``
    preserves the input's height and width, while each ``DownBlock`` halves
    the spatial dimensions and doubles the number of channels.

    Parameters
    ----------
    in_chans : int
        The number of input channels the encoder should take.
    start_features : int
        The number of features the first conv block should produce.
    num_blocks : int
        The number of downsampling blocks in the encoder.
    pool_style : str
        The type of pooling to use when downsampling (``"avg"`` or ``"max"``).
    lr_slope : float
        The negative slope argument to use in the ``LeakyReLU`` layers.
    kernel_size : int
        Size of the square convolutional kernel to use in the ``Conv2d``
        layers. Should be a positive, odd, int.
    max_features
        In each of the down-sampling blocks, the numbers of features is
        doubled. Optionally supplying ``max_features`` places a limit on this.


    Examples
    --------
    >>> from torch_tools import Encoder2d
    >>> model = Encoder2d(
                    in_chans=3,
                    start_features=64,
                    num_blocks=4,
                    pool_style="max",
                    lr_slope=0.123,
                    kernel_size=3,
                    max_feats=512,
                )

    """

    def __init__(
        self,
        in_chans: int,
        start_features: int,
        num_blocks: int,
        pool_style: str,
        lr_slope: float,
        kernel_size: int,
        max_feats: Optional[int] = None,
    ):
        """Build `Encoder`."""
        super().__init__(
            DoubleConvBlock(
                process_num_feats(in_chans),
                process_num_feats(start_features),
                process_negative_slope_arg(lr_slope),
                kernel_size=process_2d_kernel_size(kernel_size),
            ),
            *self._get_blocks(
                process_num_feats(start_features),
                process_u_architecture_layers(num_blocks),
                process_str_arg(pool_style),
                process_negative_slope_arg(lr_slope),
                process_2d_kernel_size(kernel_size),
                process_max_feats_arg(max_feats),
            ),
        )

    def _get_blocks(
        self,
        in_chans: int,
        num_blocks,
        pool_style: str,
        lr_slope: float,
        kernel_size: int,
        max_feats: Optional[int] = None,
    ) -> List[DownBlock]:
        """Get the encoding layers in a sequential.

        Parameters
        ----------
        in_chans : int
            The number of input channels.
        num_blocks : int
            The number of blocks in the encoder.
        pool_style : str
            The pool style to use when downsampling (``"avg"`` or ``"max"``).
        lr_slope : float
            The negative slope to use in the ``LeakyReLU`` layers.
        kernel_size : int
            Size of the square convolutional kernel to use in the ``Conv2d``
            layers. Should be a positive, odd, int.
        max_feats : int, optional
            Optional limit on the maximum number of features the down blocks
            can produce.

        Returns
        -------
        List[DownBlock]
            A list of the model's blocks.

        """
        chans = in_chans
        blocks = []
        for _ in range(num_blocks - 1):
            in_chans, out_chans = chans, chans * 2
            if max_feats is not None:
                in_chans = min(in_chans, max_feats)
                out_chans = min(out_chans, max_feats)

            blocks.append(
                DownBlock(
                    in_chans,
                    out_chans,
                    pool_style,
                    lr_slope,
                    kernel_size=kernel_size,
                )
            )
            chans *= 2
        return blocks
