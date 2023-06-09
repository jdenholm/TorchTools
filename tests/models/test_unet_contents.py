"""Test the contents of the ``UNet``."""
from itertools import product

from torch_tools import UNet


def test_in_conv_contents_with_all_relevant_args():
    """Test the contents of the ``in_conv`` with all relevent args."""
    in_channels = [1, 3, 5]
    start_feats = [16, 32]
    lr_slopes = [0.0, 0.1, 0.2]
    kernel_sizes = [1, 3, 5]

    iterator = product(in_channels, start_feats, lr_slopes, kernel_sizes)

    for in_chans, feats, lr_slope, kernel_size in iterator:
        model = UNet(
            in_chans=in_chans,
            out_chans=1,
            features_start=feats,
            lr_slope=lr_slope,
            kernel_size=kernel_size,
        )

        # Test input channels of first conv layer
        assert model.in_conv[0][0].in_channels == in_chans

        # Test both conv blocks produce the right numbers of features
        assert model.in_conv[0][0].out_channels == feats
        assert model.in_conv[1][0].out_channels == feats

        # Test the leaky relus have the right slope
        assert model.in_conv[0][2].negative_slope == lr_slope
        assert model.in_conv[1][2].negative_slope == lr_slope

        # Test the kernel sizes of the conv layers
        assert model.in_conv[0][0].kernel_size == (kernel_size, kernel_size)
        assert model.in_conv[1][0].kernel_size == (kernel_size, kernel_size)


def test_out_conv_contents():
    """Test the conents of the ``out_conv``."""
    features_start = [16, 32]
    out_channels = [1, 4, 10]

    for feats, out_chans in product(features_start, out_channels):
        model = UNet(in_chans=1, out_chans=out_chans, features_start=feats)

        assert model.out_conv.in_channels == feats
        assert model.out_conv.out_channels == out_chans
