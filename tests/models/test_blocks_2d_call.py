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
    in_channels = [3, 123, 321]
    out_channels = [3, 123, 321]
    lr_slopes = [0.0, 0.1, 0.2]
    kernel_sizes = [1, 3, 5]

    arg_iter = product(in_channels, out_channels, lr_slopes, kernel_sizes)

    for in_chans, out_chans, slope, kernel_size in arg_iter:
        block = DoubleConvBlock(
            in_chans=in_chans,
            out_chans=out_chans,
            lr_slope=slope,
            kernel_size=kernel_size,
        )
        assert block(rand(10, in_chans, 50, 100)).shape == (10, out_chans, 50, 100)


def test_res_block_call_return_shapes():
    """Test the return shapes produced by `ResBlock`."""
    in_channels = [3, 10, 50]
    kernel_sizes = [3, 5, 7]

    for in_chans, kernel_size in product(in_channels, kernel_sizes):
        block = ResidualBlock(in_chans=in_chans, kernel_size=kernel_size)
        out = block(rand(10, in_chans, 50, 100))
        assert out.shape == (10, in_chans, 50, 100)


def test_down_block_call_return_shapes():
    """Test the return shapes of `DownBlock`."""
    in_channels = [2, 4, 5]
    out_channels = [2, 4, 5]
    pools = ["max", "avg"]
    lr_slopes = [0.0, 0.1]
    kernel_sizes = [1, 3, 5]

    iterator = product(
        in_channels,
        out_channels,
        pools,
        lr_slopes,
        kernel_sizes,
    )

    for ins, outs, pool, slope, size in iterator:
        batch = rand(10, ins, 50, 100)

        block = DownBlock(ins, outs, pool, slope, size)

        assert block(batch).shape == (10, outs, 25, 50)


def test_up_block_call_return_shapes():
    """Test the return shapes produced by `UpBlock`."""
    in_channels = [3, 5, 7]
    out_channels = [3, 5, 7]
    bilinears = [True, False]
    slopes = [0.0, 0.1]
    kernels = [1, 3, 5]

    iterator = product(in_channels, out_channels, bilinears, slopes, kernels)

    for in_chans, out_chans, bilinear, lr_slope, kernel_size in iterator:
        block = UpBlock(
            in_chans=in_chans,
            out_chans=out_chans,
            bilinear=bilinear,
            lr_slope=lr_slope,
            kernel_size=kernel_size,
        )
        assert block(rand(10, in_chans, 16, 32)).shape == (10, out_chans, 32, 64)


def test_unet_upblock_call_return_shapes():
    """Test the return shapes produced by `UNetUpBlock`."""
    in_channels = [2, 4, 8]
    out_channels = [2, 4, 8]
    bilinears = [True, False]
    slopes = [0.0, 0.1]
    kernels = [1, 3, 5]

    iterator = product(in_channels, out_channels, bilinears, slopes, kernels)

    for in_chans, out_chans, bilinear, lr_slope, kernel_size in iterator:
        block = UNetUpBlock(
            in_chans,
            out_chans,
            bilinear,
            lr_slope,
            kernel_size=kernel_size,
        )

        to_upsample = rand(10, in_chans, 25, 50)
        down_features = rand(10, in_chans // 2, 50, 100)

        out = block(to_upsample, down_features)
        assert out.shape == (10, out_chans, 50, 100)
