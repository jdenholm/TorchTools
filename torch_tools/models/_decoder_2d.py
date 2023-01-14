"""Two-dimensional decoder model."""
from typing import List

from torch.nn import Sequential, Conv2d


from torch_tools.models._blocks_2d import UpBlock
from torch_tools.models._argument_processing import process_num_feats
from torch_tools.misc import divides_by_two_check

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
        Whether to use bilinear interpolation (`True`) or a `ConvTranspose2d`
        to do the upsampling.
    lr_slope : float
        The negative slope to use in the `LeakyReLU`s.

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        num_blocks: int,
        bilinear: bool,
        lr_slope: float,
    ):
        """Build `Decoder`."""
        super().__init__(
            *self._get_blocks(
                process_num_feats(in_chans),
                num_blocks,
                bilinear,
                lr_slope,
            ),
            Conv2d(
                in_channels=in_chans // (2 ** (num_blocks - 1)),
                out_channels=out_chans,
                kernel_size=1,
                stride=1,
            ),
        )

    def _get_blocks(
        self,
        in_chans: int,
        num_blocks: int,
        bilinear: bool,
        lr_slope: float,
    ) -> List[UpBlock]:
        """Get the upsampling blocks in a `Sequential`.

        Parameters
        ----------
        in_chans : int
            The number of blocks the model should take.
        num_blocks : int
            The number of blocks in the model.
        bilinear : bool
            Whether to use bilinear interpolation (`True`) or a
            `ConvTranspose2d` to upsample.
        lr_slope : float
            Negative slope to use in the `LeakReLU`s.

        Returns
        -------
        blocks : List[UpBlock]
            A list of the upsampling layers to include in the decoder.

        """
        chans = in_chans
        blocks = []
        for _ in range(num_blocks - 1):

            divides_by_two_check(chans)

            blocks.append(
                UpBlock(
                    process_num_feats(chans),
                    process_num_feats(chans // 2),
                    bilinear,
                    lr_slope,
                )
            )
            chans //= 2
        return blocks
