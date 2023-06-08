"""Tests for the ``UNet``call method."""
from itertools import product

from torch import rand  # pylint:disable=no-name-in-module

from torch_tools import UNet


def test_unet_call():
    """Test the call method of the ``UNet``."""
    in_chans = [1, 2, 3]
    out_chans = [1, 2, 3]
    feats_start = [8, 16]
    num_layers = [2, 3]
    pools = ["avg", "max"]
    bilinear = [True, False]
    lr_slope = [0.0, 0.1]
    kernel_size = [3, 5]

    iterator = product(
        in_chans,
        out_chans,
        feats_start,
        num_layers,
        pools,
        bilinear,
        lr_slope,
        kernel_size,
    )

    for ins, outs, feats, layers, pool, bilin, slope, size in iterator:
        model = UNet(
            in_chans=ins,
            out_chans=outs,
            features_start=feats,
            pool_style=pool,
            num_layers=layers,
            bilinear=bilin,
            lr_slope=slope,
            kernel_size=size,
        )

        assert model(rand(10, ins, 25, 50)).shape == (10, outs, 25, 50)
