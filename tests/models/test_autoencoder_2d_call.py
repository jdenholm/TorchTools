"""Test the call behaviour of ``torch_tools.AutoEncoder2d``."""
from itertools import product

from torch import rand  # pylint: disable=no-name-in-module


from torch_tools import AutoEncoder2d


def test_autoencoder_2d_return_shapes():
    """Test the shapes of the output of ``AutoEncoder2d``."""
    in_channels = [1, 123]
    out_channels = [1, 3]
    num_layers = [2, 3]
    features_start = [16, 32]
    slopes = [0.0, 0.1]
    pools = ["avg", "max"]
    bilinear = [True, False]
    kernel_size = [1, 3, 5]

    iterator = product(
        in_channels,
        out_channels,
        num_layers,
        features_start,
        slopes,
        pools,
        bilinear,
        kernel_size,
    )

    for ins, outs, layers, feats, slope, pool, bilin, size in iterator:
        model = AutoEncoder2d(
            in_chans=ins,
            out_chans=outs,
            num_layers=layers,
            features_start=feats,
            lr_slope=slope,
            pool_style=pool,
            bilinear=bilin,
            kernel_size=size,
        )

        assert model(rand(10, ins, 16, 32)).shape == (10, outs, 16, 32)
