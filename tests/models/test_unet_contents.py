"""Test the contents of the ``UNet``."""
from itertools import product

from torch.nn import Module, LeakyReLU, Conv2d, AvgPool2d, MaxPool2d
from torch.nn import ConvTranspose2d, Sequential, Upsample, BatchNorm2d

from torch_tools import UNet
from torch_tools.models._blocks_2d import (
    DownBlock,
    DoubleConvBlock,
    ConvBlock,
    ConvResBlock,
    ResidualBlock,
    UNetUpBlock,
)


# pylint: disable=cell-var-from-loop


def leaky_relu_slope_check(layer: Module, slope: float):
    """Check all ``LeakyRelU``s have the correct negative slope."""
    if isinstance(layer, LeakyReLU):
        assert layer.negative_slope == slope


def kernel_size_check(layer: Module, size: int):
    """Check the kernel size in ``Conv2d`` blocks."""
    if isinstance(layer, Conv2d):
        assert layer.kernel_size == (size, size)


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


def test_leaky_relu_slopes():
    """Test the negative slopes of the ``LeakyReLU`` layers."""
    for slope in [0.0, 0.1, 0.2]:
        model = UNet(in_chans=1, out_chans=1, lr_slope=slope)
        model.apply(lambda x: leaky_relu_slope_check(x, slope))


def test_kernel_sizes_of_relevant_components():
    """Test the kernel sizes of the relevant components."""
    for kernel_size in [1, 3, 5, 7]:
        model = UNet(in_chans=1, out_chans=1, kernel_size=kernel_size)

        model.in_conv.apply(lambda x: kernel_size_check(x, kernel_size))
        model.down_blocks.apply(lambda x: kernel_size_check(x, kernel_size))
        model.up_blocks.apply(lambda x: kernel_size_check(x, kernel_size))


def test_down_block_pool_type():
    """Test the pool type of the ``DownBlock`` layers."""
    model = UNet(in_chans=1, out_chans=1, pool_style="avg")
    for block in model.down_blocks.children():
        if isinstance(block, DownBlock):
            assert isinstance(block[0], AvgPool2d)
            assert block[0].kernel_size == 2
            assert block[0].stride == 2

    model = UNet(in_chans=1, out_chans=1, pool_style="max")
    for block in model.down_blocks.children():
        if isinstance(block, DownBlock):
            assert isinstance(block[0], MaxPool2d)
            assert block[0].kernel_size == 2
            assert block[0].stride == 2


def test_upsample_contents():
    """Test the upsampling methods in the ``UpBlocks``."""
    model = UNet(in_chans=1, out_chans=1, bilinear=False)
    for block in model.up_blocks:
        assert isinstance(block.upsample, ConvTranspose2d)
        assert block.upsample.kernel_size == (2, 2)
        assert block.upsample.stride == (2, 2)

    model = UNet(in_chans=1, out_chans=1, bilinear=True)
    for block in model.up_blocks:
        assert isinstance(block.upsample, Sequential)
        assert isinstance(block.upsample[0], Upsample)
        assert isinstance(block.upsample[1], Conv2d)


def test_down_blocks_channels():
    """Test the number of channels at each point in the down blocks."""
    for feats in [8, 16, 32]:
        model = UNet(in_chans=1, out_chans=1, features_start=feats)
        down_blocks = model.down_blocks

        for block in down_blocks:
            # Test the first conv block of the double conv block
            assert block[1][0][0].in_channels == feats
            assert block[1][0][0].out_channels == feats * 2

            # Test the second conv block of the double conv block
            assert block[1][0][0].in_channels == feats
            assert block[1][1][0].out_channels == feats * 2

            feats *= 2


def test_up_blocks_channels_without_blinear_interp():
    """Test the number of channels in the up blocks with no bilinear interp."""
    for feats in [8, 16, 32]:
        model = UNet(
            in_chans=1,
            out_chans=1,
            features_start=feats,
            bilinear=False,
            num_layers=3,
        )
        up_blocks = model.up_blocks

        feats = feats * 2 ** (2)

        for block in up_blocks:
            # Test the first block in the double conv block
            assert block.conv_block[0][0].in_channels == feats
            assert block.conv_block[0][0].out_channels == feats // 2

            # Test the second block in the double conv block
            assert block.conv_block[1][0].in_channels == feats // 2
            assert block.conv_block[1][0].out_channels == feats // 2

            feats //= 2


def test_up_blocks_channels_with_blinear_interp():
    """Test the number of channels in the up blocks with bilinear interp."""
    for feats in [8, 16, 32]:
        model = UNet(
            in_chans=1,
            out_chans=1,
            features_start=feats,
            bilinear=True,
            num_layers=3,
        )
        up_blocks = model.up_blocks

        feats = feats * 2 ** (2)

        for block in up_blocks:
            # Test the output conv of the upsample layer
            # If you are confused why the output channels are halved,
            # remember this bit acts on a Tensor that will be concatenated
            # with down features
            assert block.upsample[1].in_channels == feats
            assert block.upsample[1].out_channels == feats // 2

            # Test the first block in the double conv block
            assert block.conv_block[0][0].in_channels == feats
            assert block.conv_block[0][0].out_channels == feats // 2

            # Test the second block in the double conv block
            assert block.conv_block[1][0].in_channels == feats // 2
            assert block.conv_block[1][0].out_channels == feats // 2

            feats //= 2


def test_in_conv_contents_with_double_conv_block():
    """Test the contents of the ``in_conv`` with double conv blocks."""
    model = UNet(
        in_chans=123,
        out_chans=1,
        features_start=321,
        block_style="double_conv",
        lr_slope=0.123456,
    )

    in_conv = model.in_conv

    assert isinstance(in_conv, DoubleConvBlock)

    assert isinstance(in_conv[0], ConvBlock)
    assert isinstance(in_conv[0][0], Conv2d)
    assert isinstance(in_conv[0][1], BatchNorm2d)
    assert isinstance(in_conv[0][2], LeakyReLU)

    assert in_conv[0][0].in_channels == 123
    assert in_conv[0][0].out_channels == 321
    assert in_conv[0][1].num_features == 321
    assert in_conv[0][2].negative_slope == 0.123456

    assert isinstance(in_conv[0], ConvBlock)
    assert isinstance(in_conv[1][0], Conv2d)
    assert isinstance(in_conv[1][1], BatchNorm2d)
    assert isinstance(in_conv[1][2], LeakyReLU)

    assert in_conv[1][0].in_channels == 321
    assert in_conv[1][0].out_channels == 321
    assert in_conv[1][1].num_features == 321
    assert in_conv[1][2].negative_slope == 0.123456


def test_in_conv_contents_with_conv_res_block():
    """Test the contents of the ``in_conv`` with conv res blocks."""
    model = UNet(
        in_chans=123,
        out_chans=1,
        features_start=321,
        block_style="conv_res",
        lr_slope=0.123456,
    )

    in_conv = model.in_conv

    assert isinstance(in_conv, ConvResBlock)
    assert isinstance(in_conv[0], ConvBlock)
    assert isinstance(in_conv[0][0], Conv2d)
    assert isinstance(in_conv[0][1], BatchNorm2d)
    assert isinstance(in_conv[0][2], LeakyReLU)

    assert in_conv[0][0].in_channels == 123
    assert in_conv[0][0].out_channels == 321
    assert in_conv[0][1].num_features == 321
    assert in_conv[0][2].negative_slope == 0.123456

    assert isinstance(in_conv[1], ResidualBlock)
    assert isinstance(in_conv[1].first_conv, ConvBlock)
    assert isinstance(in_conv[1].first_conv[0], Conv2d)
    assert isinstance(in_conv[1].first_conv[1], BatchNorm2d)
    assert isinstance(in_conv[1].first_conv[2], LeakyReLU)

    assert in_conv[1].first_conv[0].in_channels == 321
    assert in_conv[1].first_conv[0].out_channels == 321
    assert in_conv[1].first_conv[1].num_features == 321
    assert in_conv[1].first_conv[2].negative_slope == 0.0

    assert isinstance(in_conv[1].second_conv, ConvBlock)
    assert isinstance(in_conv[1].second_conv[0], Conv2d)
    assert isinstance(in_conv[1].second_conv[1], BatchNorm2d)

    assert in_conv[1].second_conv[0].in_channels == 321
    assert in_conv[1].second_conv[0].out_channels == 321
    assert in_conv[1].second_conv[1].num_features == 321


def test_unet_down_block_contents_with_double_conv_blocks():
    """Test the contents of the down blocks with double conv style."""
    model = UNet(
        in_chans=1,
        out_chans=1,
        features_start=16,
        block_style="double_conv",
        lr_slope=0.123456,
        pool_style="max",
    )

    down_blocks = model.down_blocks

    in_chans, out_chans = 16, 32

    for block in down_blocks.children():
        assert isinstance(block, DownBlock)
        assert isinstance(block[0], MaxPool2d)

        assert isinstance(block[1], DoubleConvBlock)
        assert isinstance(block[1][0], ConvBlock)
        assert isinstance(block[1][0][0], Conv2d)
        assert isinstance(block[1][0][1], BatchNorm2d)
        assert isinstance(block[1][0][2], LeakyReLU)

        assert block[1][0][0].in_channels == in_chans
        assert block[1][0][0].out_channels == out_chans
        assert block[1][0][1].num_features == out_chans
        assert block[1][0][2].negative_slope == 0.123456

        assert isinstance(block[1][1], ConvBlock)
        assert isinstance(block[1][1][0], Conv2d)
        assert isinstance(block[1][1][1], BatchNorm2d)
        assert isinstance(block[1][1][2], LeakyReLU)

        assert block[1][1][0].in_channels == out_chans
        assert block[1][1][0].out_channels == out_chans
        assert block[1][1][1].num_features == out_chans
        assert block[1][1][2].negative_slope == 0.123456

        in_chans *= 2
        out_chans *= 2


def test_unet_down_block_contents_with_conv_res_blocks():
    """Test the contents of the down blocks with conv res style."""
    model = UNet(
        in_chans=1,
        out_chans=1,
        features_start=16,
        block_style="conv_res",
        lr_slope=0.123456,
        pool_style="max",
    )

    down_blocks = model.down_blocks

    in_chans, out_chans = 16, 32

    for block in down_blocks.children():
        assert isinstance(block, DownBlock)
        assert isinstance(block[0], MaxPool2d)

        assert isinstance(block[1], ConvResBlock)
        assert isinstance(block[1][0], ConvBlock)
        assert isinstance(block[1][0][0], Conv2d)
        assert isinstance(block[1][0][1], BatchNorm2d)
        assert isinstance(block[1][0][2], LeakyReLU)

        assert block[1][0][0].in_channels == in_chans
        assert block[1][0][0].out_channels == out_chans
        assert block[1][0][1].num_features == out_chans
        assert block[1][0][2].negative_slope == 0.123456

        assert isinstance(block[1][1], ResidualBlock)
        assert isinstance(block[1][1].first_conv, ConvBlock)
        assert isinstance(block[1][1].first_conv[0], Conv2d)
        assert isinstance(block[1][1].first_conv[1], BatchNorm2d)
        assert isinstance(block[1][1].first_conv[2], LeakyReLU)

        assert block[1][1].first_conv[0].in_channels == out_chans
        assert block[1][1].first_conv[0].out_channels == out_chans
        assert block[1][1].first_conv[1].num_features == out_chans
        assert block[1][1].first_conv[2].negative_slope == 0.0

        assert isinstance(block[1][1].second_conv, ConvBlock)
        assert isinstance(block[1][1].second_conv[0], Conv2d)
        assert isinstance(block[1][1].second_conv[1], BatchNorm2d)

        assert block[1][1].second_conv[0].in_channels == out_chans
        assert block[1][1].second_conv[0].out_channels == out_chans
        assert block[1][1].second_conv[1].num_features == out_chans

        in_chans *= 2
        out_chans *= 2


def test_unet_up_block_contents_with_double_conv_blocks():
    """Test the contents of the up blocks with double conv style."""
    model = UNet(
        in_chans=1,
        out_chans=1,
        features_start=16,
        block_style="double_conv",
        lr_slope=0.123456,
        pool_style="max",
        num_layers=4,
    )

    down_blocks = model.up_blocks

    in_chans = 16 * (2**3)
    out_chans = in_chans // 2

    for block in down_blocks.children():
        assert isinstance(block.conv_block, DoubleConvBlock)
        assert isinstance(block.conv_block[0], ConvBlock)
        assert isinstance(block.conv_block[0][0], Conv2d)
        assert isinstance(block.conv_block[0][1], BatchNorm2d)
        assert isinstance(block.conv_block[0][2], LeakyReLU)

        assert block.conv_block[0][0].in_channels == in_chans
        assert block.conv_block[0][0].out_channels == out_chans
        assert block.conv_block[0][1].num_features == out_chans
        assert block.conv_block[0][2].negative_slope == 0.123456

        assert isinstance(block.conv_block[1], ConvBlock)
        assert isinstance(block.conv_block[1][0], Conv2d)
        assert isinstance(block.conv_block[1][1], BatchNorm2d)
        assert isinstance(block.conv_block[1][2], LeakyReLU)

        assert block.conv_block[1][0].in_channels == out_chans
        assert block.conv_block[1][0].out_channels == out_chans
        assert block.conv_block[1][1].num_features == out_chans
        assert block.conv_block[1][2].negative_slope == 0.123456

        in_chans //= 2
        out_chans //= 2


def test_unet_up_blocks_contents_with_conv_res_blocks():
    """Test the contents of the up blocks with conv res style."""
    model = UNet(
        in_chans=1,
        out_chans=1,
        features_start=16,
        block_style="conv_res",
        lr_slope=0.123456,
        pool_style="max",
        num_layers=4,
    )

    up_blocks = model.up_blocks

    in_chans = 16 * 2**3
    out_chans = in_chans // 2

    for block in up_blocks.children():
        assert isinstance(block, UNetUpBlock)

        assert isinstance(block.conv_block, ConvResBlock)
        assert isinstance(block.conv_block[0], ConvBlock)
        assert isinstance(block.conv_block[0][0], Conv2d)
        assert isinstance(block.conv_block[0][1], BatchNorm2d)
        assert isinstance(block.conv_block[0][2], LeakyReLU)

        assert block.conv_block[0][0].in_channels == in_chans
        assert block.conv_block[0][0].out_channels == out_chans
        assert block.conv_block[0][1].num_features == out_chans
        assert block.conv_block[0][2].negative_slope == 0.123456

        assert isinstance(block.conv_block[1], ResidualBlock)
        assert isinstance(block.conv_block[1].first_conv, ConvBlock)
        assert isinstance(block.conv_block[1].first_conv[0], Conv2d)
        assert isinstance(block.conv_block[1].first_conv[1], BatchNorm2d)
        assert isinstance(block.conv_block[1].first_conv[2], LeakyReLU)

        assert block.conv_block[1].first_conv[0].in_channels == out_chans
        assert block.conv_block[1].first_conv[0].out_channels == out_chans
        assert block.conv_block[1].first_conv[1].num_features == out_chans
        assert block.conv_block[1].first_conv[2].negative_slope == 0.0

        assert isinstance(block.conv_block[1].second_conv, ConvBlock)
        assert isinstance(block.conv_block[1].second_conv[0], Conv2d)
        assert isinstance(block.conv_block[1].second_conv[1], BatchNorm2d)

        assert block.conv_block[1].second_conv[0].in_channels == out_chans
        assert block.conv_block[1].second_conv[0].out_channels == out_chans
        assert block.conv_block[1].second_conv[1].num_features == out_chans

        in_chans //= 2
        out_chans //= 2
