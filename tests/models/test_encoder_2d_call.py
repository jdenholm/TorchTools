"""Test the call behaviour of `torch_tools.models._encoder_2d.Encoder2d`."""
from itertools import product
from torch import rand  # pylint: disable=no-name-in-module

from torch_tools import Encoder2d


def test_encoder_2d_return_shapes_with_brute_force_arg_combos():
    """Test the return shapes with all argument combos."""

    in_chans = [1, 2, 3]
    start_features = [8, 16]
    num_blocks = [2, 3]
    pool_style = ["avg", "max"]
    lr_slope = [0.0, 0.1]
    kernel_sizes = [1, 3, 5]
    max_features = [None, 64, 128]
    block_styles = ["double_conv", "conv_res"]

    iterator = product(
        in_chans,
        start_features,
        num_blocks,
        pool_style,
        lr_slope,
        kernel_sizes,
        max_features,
        block_styles,
    )

    for (
        in_chans,
        feats,
        blocks,
        pool,
        slope,
        kernel_size,
        max_feats,
        block_style,
    ) in iterator:
        model = Encoder2d(
            in_chans,
            feats,
            blocks,
            pool,
            slope,
            kernel_size,
            max_feats=max_feats,
            block_style=block_style,
        )

        out = model(rand(1, in_chans, 32, 64))

        assert out.shape == (
            1,
            feats * (2 ** (blocks - 1)),
            32 // (2 ** (blocks - 1)),
            64 // (2 ** (blocks - 1)),
        )
