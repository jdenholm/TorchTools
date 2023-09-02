"""Two-dimensional decoder model."""
from typing import List, Optional

from torch.nn import Sequential, Conv2d


from torch_tools.models._blocks_2d import UpBlock
from torch_tools.models._argument_processing import process_num_feats
from torch_tools.models._argument_processing import process_boolean_arg
from torch_tools.models._argument_processing import process_negative_slope_arg
from torch_tools.models._argument_processing import process_2d_kernel_size
from torch_tools.models._argument_processing import process_optional_feats_arg

# pylint: disable=too-many-arguments


class Decoder2d(Sequential):
    """Simple decoder model for image-like inputs.

    Parameters
    ----------
    in_chans : int
        The number of input channels the model should take.
    out_chans : int
        The number of output channels the decoder should produce.
    num_blocks : int
        The number of blocks to include in the decoder.
    bilinear : bool
        Whether to use bilinear interpolation (``True``) or a
        ``ConvTranspose2d`` to do the upsampling.
    lr_slope : float
        The negative slope to use in the ``LeakyReLU`` layers.
    kernel_size : int
        The size of the square convolutional kernel in the ``Conv2d`` layers.
        Should be an odd, positive, int.
    min_up_feats : int, optional
        The minimum numbers features the up-sampling blocks are allowed to
        produce.

    Examples
    --------
    >>> from torch_tools import Decoder2d
    >>> model = Decoder2d(
                    in_chans=128,
                    out_chans=3,
                    num_blocks=4,
                    bilinear=False,
                    lr_slope=0.123,
                    kernel_size=3,
                )

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        num_blocks: int,
        bilinear: bool,
        lr_slope: float,
        kernel_size: int,
        min_up_feats: Optional[int] = None,
    ):
        """Build `Decoder`."""
        super().__init__(
            *self._get_blocks(
                process_num_feats(in_chans),
                process_num_feats(num_blocks),
                process_boolean_arg(bilinear),
                process_negative_slope_arg(lr_slope),
                process_2d_kernel_size(kernel_size),
                min_feats=process_optional_feats_arg(min_up_feats),
            ),
            Conv2d(
                in_channels=self._channel_size_check(
                    in_chans,
                    num_blocks,
                    min_feats=process_optional_feats_arg(min_up_feats),
                ),
                out_channels=process_num_feats(out_chans),
                kernel_size=1,
                stride=1,
            ),
        )

    @staticmethod
    def _channel_size_check(
        in_chans: int,
        num_blocks: int,
        min_feats: Optional[int] = None,
    ) -> int:
        """Check ``in_chans`` can be halved ``num_layers - 1`` times.

        Parameters
        ----------
        in_chans : int
            The number of inputs channels the model should take.
        num_blocks : int
            The number of layers in the model.
        min_feats : int, optional
            Min number of out feats the up-sampling layers can produce.

        Returns
        -------
        chans : int
            The number of channels the up blocks should produce.

        Raises
        ------
        ValueError
            If ``in_chans`` cannot be divided by 2 ``num_blocks - 1`` times.

        """
        chans = in_chans
        for _ in range(num_blocks - 1):
            if (chans % 2) != 0 and (chans != min_feats):
                msg = f"'in_chans' value {in_chans} can't be halved "
                msg += f"{num_blocks - 1} times."
                raise ValueError(msg)
            chans //= 2

            if min_feats is not None:
                chans = max(min_feats, chans)

        return chans

    def _get_blocks(
        self,
        in_chans: int,
        num_blocks: int,
        bilinear: bool,
        lr_slope: float,
        kernel_size: int,
        min_feats: Optional[int] = None,
    ) -> List[UpBlock]:
        """Get the upsampling blocks in a ``Sequential``.

        Parameters
        ----------
        in_chans : int
            The number of blocks the model should take.
        num_blocks : int
            The number of blocks in the model.
        bilinear : bool
            Whether to use bilinear interpolation (``True``) or a
            ``ConvTranspose2d`` to upsample.
        lr_slope : float
            Negative slope to use in the ``LeakReLU`` layers.
        kernel_size : int
            The size of the square convolutional kernel in the ``Conv2d``
            layers. Should be an odd, positive, int.
        min_feats : int, optional
            Min number of out feats the up-sampling layers can produce.

        Returns
        -------
        blocks : List[UpBlock]
            A list of the upsampling layers to include in the decoder.

        """
        self._channel_size_check(in_chans, num_blocks, min_feats=min_feats)
        chans = in_chans
        blocks = []
        for _ in range(num_blocks - 1):
            if min_feats is None:
                in_chans, out_chans = chans, chans // 2
            else:
                in_chans = max(min_feats, chans)
                out_chans = max(min_feats, chans // 2)

            blocks.append(
                UpBlock(
                    process_num_feats(in_chans),
                    process_num_feats(out_chans),
                    bilinear,
                    lr_slope,
                    kernel_size,
                )
            )
            chans //= 2
        return blocks
