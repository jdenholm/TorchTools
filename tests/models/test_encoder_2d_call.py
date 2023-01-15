"""Test the call behaviour of `torch_tools.models._encoder_2d.Encoder2d`."""
from torch import rand  # pylint: disable=no-name-in-module

from torch_tools import Encoder2d


def test_encoder_2d_return_channels_with_max_pool():
    """Test the number of channels returned by the call method.

    Notes
    -----
    Aside from the first convolutional block, each block in the encoder
    should halve the 'image' size and double the number of channels.


    The number of output channels should be
    (2 ** num_layers - 1) * start_features.

    """
    block = Encoder2d(
        in_chans=3,
        start_features=32,
        num_blocks=4,
        pool_style="max",
        lr_slope=0.1,
    )

    assert block(rand(10, 3, 256, 256)).shape[1] == (32 * 2 ** (4 - 1))

    block = Encoder2d(
        in_chans=3,
        start_features=64,
        num_blocks=6,
        pool_style="max",
        lr_slope=0.1,
    )

    assert block(rand(10, 3, 256, 256)).shape[1] == (64 * 2 ** (6 - 1))


def test_encoder_2d_return_channels_with_avgavg_pool():
    """Test the number of channels returned by the call method.

    Notes
    -----
    Aside from the first convolutional block, each block in the encoder
    should halve the 'image' size and double the number of channels.


    The number of output channels should be
    (2 ** num_layers - 1) * start_features.

    """
    block = Encoder2d(
        in_chans=3,
        start_features=32,
        num_blocks=4,
        pool_style="avg",
        lr_slope=0.1,
    )

    assert block(rand(10, 3, 256, 256)).shape[1] == (32 * 2 ** (4 - 1))

    block = Encoder2d(
        in_chans=3,
        start_features=64,
        num_blocks=6,
        pool_style="avg",
        lr_slope=0.1,
    )

    assert block(rand(10, 3, 256, 256)).shape[1] == (64 * 2 ** (6 - 1))


def test_encoder_2d_return_shapes_with_max_pool():
    """Test the size of the output 'images' returned by the call method.

    Notes
    -----
    Aside from the first convolutional block, each block in the encoder
    should halve the 'image' size and double the number of channels.

    """
    block = Encoder2d(
        in_chans=3,
        start_features=32,
        num_blocks=4,
        pool_style="max",
        lr_slope=0.1,
    )

    _, _, height, width = block(rand(10, 3, 256, 512)).shape
    assert height == 32, "Return height should be 32."
    assert width == 64, "Return width should be 64."

    block = Encoder2d(
        in_chans=3,
        start_features=32,
        num_blocks=6,
        pool_style="max",
        lr_slope=0.1,
    )

    _, _, height, width = block(rand(10, 3, 512, 256)).shape
    assert height == 16, "Return height should be 32."
    assert width == 8, "Return width should be 64."


def test_encoder_2d_return_shapes_with_avg_pool():
    """Test the size of the output 'images' returned by the call method.

    Notes
    -----
    Aside from the first convolutional block, each block in the encoder
    should halve the 'image' size and double the number of channels.

    """
    block = Encoder2d(
        in_chans=3,
        start_features=32,
        num_blocks=4,
        pool_style="avg",
        lr_slope=0.1,
    )

    _, _, height, width = block(rand(10, 3, 256, 512)).shape
    assert height == 32, "Return height should be 32."
    assert width == 64, "Return width should be 64."

    block = Encoder2d(
        in_chans=3,
        start_features=32,
        num_blocks=6,
        pool_style="avg",
        lr_slope=0.1,
    )

    _, _, height, width = block(rand(10, 3, 512, 256)).shape
    assert height == 16, "Return height should be 32."
    assert width == 8, "Return width should be 64."
