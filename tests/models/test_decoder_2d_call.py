"""Test the call behaviour in `Decoder2d` is as expected."""
from itertools import product

from torch import rand  # pylint: disable=no-name-in-module

from torch_tools.models import Decoder2d


def test_decoder_2d_returns_images_of_the_right_shape():
    """Test the dimensionality of the images returned by `Decoder2d`."""
    in_chans = [512, 256]
    out_chans = [1, 2]
    num_blocks = [2, 3, 4, 5]
    biliner = [True, False]
    lr_slopes = [0.0, 0.1]
    kernel_size = [1, 3, 5]
    min_up_feats = [None, 1, 3]
    block_styles = ["double_conv", "conv_res"]

    iterator = product(
        in_chans,
        out_chans,
        num_blocks,
        biliner,
        lr_slopes,
        kernel_size,
        min_up_feats,
        block_styles,
    )

    for ins, outs, blocks, bilin, slope, size, min_feats, block_style in iterator:
        decoder = Decoder2d(
            in_chans=ins,
            out_chans=outs,
            num_blocks=blocks,
            bilinear=bilin,
            lr_slope=slope,
            kernel_size=size,
            min_up_feats=min_feats,
            block_style=block_style,
        )

        mini_batch = rand(10, ins, 8, 4)
        assert decoder(mini_batch).shape == (
            10,
            outs,
            (8 * 2 ** (blocks - 1)),
            (4 * 2 ** (blocks - 1)),
        )
