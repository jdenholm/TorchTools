"""Test the call behaviour of ``torch_tools.SimpleConvNet2d``."""
from itertools import product

from torch import rand  # pylint: disable=no-name-in-module

from torch_tools import SimpleConvNet2d


def test_simple_conv_net_2d_call():
    """Test the model's forward call method."""
    ins = [1, 3]
    outs = [2, 4, 6]
    features_start = [16, 32]
    blocks = [2, 3]
    down_pools = ["avg", "max"]
    adaptive_pool = ["avg", "max", "avg-max-concat"]
    lr_slopes = [0.0, 0.1]
    sizes = [1, 3, 5]
    block_styles = ["double_conv", "conv_res"]

    iterator = product(
        ins,
        outs,
        features_start,
        blocks,
        down_pools,
        adaptive_pool,
        lr_slopes,
        sizes,
        block_styles,
    )

    for (
        in_chans,
        out_feats,
        feats,
        num_blocks,
        d_pool,
        a_pool,
        slope,
        size,
        block_style,
    ) in iterator:
        model = SimpleConvNet2d(
            in_chans,
            out_feats,
            features_start=feats,
            num_blocks=num_blocks,
            downsample_pool=d_pool,
            adaptive_pool=a_pool,
            lr_slope=slope,
            kernel_size=size,
            block_style=block_style,
        )

        assert model(rand(10, in_chans, 100, 100)).shape == (10, out_feats)
