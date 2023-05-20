"""Tests for the call methods of blocks in `torch_tools.models._blocks_2d`."""
from itertools import product


from torch import rand  # pylint: disable=no-name-in-module

from torch_tools.models._blocks_2d import ConvBlock, DoubleConvBlock, ResidualBlock
from torch_tools.models._blocks_2d import DownBlock, UpBlock, UNetUpBlock


def test_conv_block_call_return_shapes():
    """Test the return shapes with various combinations of all arguments."""
    in_channels = [3, 123, 321]
    out_channels = [3, 123, 321]
    batch_norms = [True, False]
    leaky_relus = [True, False]
    negative_slopes = [0.0, 0.1]
    kernels = [3, 5, 7]

    arg_iter = product(
        in_channels,
        out_channels,
        batch_norms,
        leaky_relus,
        negative_slopes,
        kernels,
    )

    for in_chans, out_chans, bnorm, leaky, slope, kernel_size in arg_iter:
        block = ConvBlock(
            in_chans=in_chans,
            out_chans=out_chans,
            batch_norm=bnorm,
            leaky_relu=leaky,
            lr_slope=slope,
            kernel_size=kernel_size,
        )

        assert block(rand(10, in_chans, 50, 100)).shape == (10, out_chans, 50, 100)


def test_double_conv_block_call_return_shapes():
    """Test the return shapes produced by `DoubleConvBlock` are correct."""
    block = DoubleConvBlock(in_chans=123, out_chans=321, lr_slope=0.1)
    assert block(rand(10, 123, 50, 100)).shape == (10, 321, 50, 100)

    block = DoubleConvBlock(in_chans=111, out_chans=222, lr_slope=0.1)
    assert block(rand(10, 111, 50, 100)).shape == (10, 222, 50, 100)


def test_res_block_call_return_shapes():
    """Test the return shapes produced by `ResBlock`."""
    block = ResidualBlock(in_chans=123)
    assert block(rand(10, 123, 50, 100)).shape == (10, 123, 50, 100)

    block = ResidualBlock(in_chans=111)
    assert block(rand(10, 111, 50, 100)).shape == (10, 111, 50, 100)


def test_down_block_call_return_shapes():
    """Test the return shapes of `DownBlock`."""
    block = DownBlock(in_chans=3, out_chans=8, pool="max", lr_slope=0.1)
    assert block(rand(10, 3, 50, 100)).shape == (10, 8, 25, 50)

    block = DownBlock(in_chans=3, out_chans=3, pool="avg", lr_slope=0.1)
    assert block(rand(10, 3, 100, 50)).shape == (10, 3, 50, 25)

    # Test with odd image sizes
    block = DownBlock(in_chans=3, out_chans=3, pool="max", lr_slope=0.1)
    assert block(rand(10, 3, 101, 51)).shape == (10, 3, 50, 25)


def test_up_block_call_return_shapes():
    """Test the return shapes produced by `UpBlock`."""
    block = UpBlock(in_chans=3, out_chans=8, bilinear=True, lr_slope=0.1)
    assert block(rand(10, 3, 16, 32)).shape == (10, 8, 32, 64)

    block = UpBlock(in_chans=3, out_chans=3, bilinear=False, lr_slope=0.1)
    assert block(rand(10, 3, 128, 256)).shape == (10, 3, 256, 512)


def test_unet_upblock_call_return_shapes():
    """Test the return shapes produced by `UNetUpBlock`."""
    block = UNetUpBlock(in_chans=4, out_chans=5, bilinear=False, lr_slope=0.1)
    to_upsample = rand(10, 4, 25, 50)
    down_features = rand(10, 2, 50, 100)
    assert block(to_upsample, down_features).shape == (10, 5, 50, 100)

    block = UNetUpBlock(in_chans=2, out_chans=12, bilinear=False, lr_slope=0.1)
    to_upsample = rand(10, 2, 30, 30)
    down_features = rand(10, 1, 50, 50)
    assert block(to_upsample, down_features).shape == (10, 12, 50, 50)
