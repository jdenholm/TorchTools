"""Test the contents of `torch_tools.models._encoder_2d.Encoder2d`."""

from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, MaxPool2d, AvgPool2d
from torch.nn import Module

from torch_tools import Encoder2d

from torch_tools.models._blocks_2d import DoubleConvBlock, ConvBlock
from torch_tools.models._blocks_2d import DownBlock, ConvResBlock, ResidualBlock


def kernel_size_check(layer: Module, kernel_size: int):
    """If ``layer`` is a ``Conv2d``, check its ``kernel_size``."""
    if isinstance(layer, Conv2d):
        assert layer.kernel_size == (kernel_size, kernel_size)


def test_encoder_2d_total_number_of_blocks():
    """Test the number of blocks in the encoder."""
    encoder = Encoder2d(
        in_chans=123,
        start_features=64,
        num_blocks=4,
        pool_style="avg",
        lr_slope=0.123,
        kernel_size=3,
    )
    assert len(encoder) == 4, "Wrong number of blocks in the encoder."

    encoder = Encoder2d(
        in_chans=123,
        start_features=111,
        num_blocks=7,
        pool_style="avg",
        lr_slope=0.123,
        kernel_size=3,
    )
    assert len(encoder) == 7, "Wrong number of blocks in the encoder."


def test_encoder_2d_double_conv_contents():
    """Test the contents of the double conv layer."""
    encoder = Encoder2d(
        in_chans=123,
        start_features=111,
        num_blocks=4,
        pool_style="avg",
        lr_slope=0.123,
        kernel_size=3,
    )

    double_conv = encoder[0]
    assert isinstance(double_conv, DoubleConvBlock)

    first_conv_block, second_conv_block = double_conv[0], double_conv[1]
    # Test the first conv block
    assert isinstance(first_conv_block, ConvBlock)
    assert isinstance(first_conv_block[0], Conv2d)
    assert isinstance(first_conv_block[1], BatchNorm2d)
    assert isinstance(first_conv_block[2], LeakyReLU)

    assert first_conv_block[0].in_channels == 123
    assert first_conv_block[0].out_channels == 111
    assert first_conv_block[1].num_features == 111
    assert first_conv_block[2].negative_slope == 0.123

    # Test the second conv block
    assert isinstance(second_conv_block, ConvBlock)
    assert isinstance(second_conv_block[0], Conv2d)
    assert isinstance(second_conv_block[1], BatchNorm2d)
    assert isinstance(second_conv_block[2], LeakyReLU)

    assert second_conv_block[0].in_channels == 111
    assert second_conv_block[0].out_channels == 111
    assert second_conv_block[1].num_features == 111
    assert second_conv_block[2].negative_slope == 0.123


def test_encoder_2d_pool_type_with_avg_pool():
    """Test the pooling layers are correctly set to average."""
    encoder = Encoder2d(
        in_chans=123,
        start_features=111,
        num_blocks=3,
        pool_style="avg",
        lr_slope=0.123,
        kernel_size=3,
    )

    # No pool in the first block, but there should be one in the others
    for down_block in list(encoder.children())[1:]:
        assert isinstance(down_block[0], AvgPool2d)


def test_encoder_2d_pool_type_with_max_pool():
    """Test the pooling layers are correctly set to max."""
    encoder = Encoder2d(
        in_chans=123,
        start_features=111,
        num_blocks=3,
        pool_style="max",
        lr_slope=0.123,
        kernel_size=3,
    )

    # No pool in the first block, but there should be one in the others
    for down_block in list(encoder.children())[1:]:
        assert isinstance(down_block[0], MaxPool2d)


def test_encoder_2d_first_down_block_contents():
    """Test the contents of the down blocks."""
    encoder = Encoder2d(
        in_chans=123,
        start_features=111,
        num_blocks=3,
        pool_style="max",
        lr_slope=0.123,
        kernel_size=3,
    )

    # Test the first `DownBlock` contents
    assert isinstance(encoder[1], DownBlock)
    assert isinstance(encoder[1][0], MaxPool2d)
    assert isinstance(encoder[1][1], DoubleConvBlock)

    # Test the first conv block of the double conv
    assert isinstance(encoder[1][1][0], ConvBlock)
    assert isinstance(encoder[1][1][0][0], Conv2d)
    assert isinstance(encoder[1][1][0][1], BatchNorm2d)
    assert isinstance(encoder[1][1][0][2], LeakyReLU)

    assert encoder[1][1][0][0].in_channels == 111
    assert encoder[1][1][0][0].out_channels == 2 * 111
    assert encoder[1][1][0][1].num_features == 2 * 111
    assert encoder[1][1][0][2].negative_slope == 0.123

    # Test the second conv block of the double conv
    assert isinstance(encoder[1][1][1], ConvBlock)
    assert isinstance(encoder[1][1][1][0], Conv2d)
    assert isinstance(encoder[1][1][1][1], BatchNorm2d)
    assert isinstance(encoder[1][1][1][2], LeakyReLU)

    assert encoder[1][1][1][0].in_channels == 2 * 111
    assert encoder[1][1][1][0].out_channels == 2 * 111
    assert encoder[1][1][1][1].num_features == 2 * 111
    assert encoder[1][1][1][2].negative_slope == 0.123


def test_encoder_2d_second_down_block_contents():
    """Test the contents of the down blocks."""
    encoder = Encoder2d(
        in_chans=123,
        start_features=111,
        num_blocks=3,
        pool_style="max",
        lr_slope=0.666,
        kernel_size=3,
    )

    # Test the second down block contents
    assert isinstance(encoder[2], DownBlock)
    assert isinstance(encoder[2][0], MaxPool2d)
    assert isinstance(encoder[2][1], DoubleConvBlock)

    # Test the contents of the first conv block of the double conv
    assert isinstance(encoder[2][1][0], ConvBlock)
    assert isinstance(encoder[2][1][0][0], Conv2d)
    assert isinstance(encoder[2][1][0][1], BatchNorm2d)
    assert isinstance(encoder[2][1][0][2], LeakyReLU)

    # Test the attributes of the contents of the first conv block
    assert encoder[2][1][0][0].in_channels == 2 * 111
    assert encoder[2][1][0][0].out_channels == 4 * 111
    assert encoder[2][1][0][1].num_features == 4 * 111
    assert encoder[2][1][0][2].negative_slope == 0.666

    # Test the contents of the second conv block of the double conv
    assert isinstance(encoder[2][1][1], ConvBlock)
    assert isinstance(encoder[2][1][1][0], Conv2d)
    assert isinstance(encoder[2][1][1][1], BatchNorm2d)
    assert isinstance(encoder[2][1][1][2], LeakyReLU)

    # Test the attributes of the contents of the second conv block
    assert encoder[2][1][1][0].in_channels == 4 * 111
    assert encoder[2][1][1][0].out_channels == 4 * 111
    assert encoder[2][1][1][1].num_features == 4 * 111
    assert encoder[2][1][1][2].negative_slope == 0.666


def test_encoder_2d_contents_with_different_kernel_sizes():
    """Test contents of the ``Conv2d`` layers with differnet kerne; sizes."""
    for kernel_size in [1, 3, 5, 7]:
        encoder = Encoder2d(
            in_chans=123,
            start_features=111,
            num_blocks=3,
            pool_style="max",
            lr_slope=0.666,
            kernel_size=kernel_size,
        )

        # pylint: disable=cell-var-from-loop
        encoder.apply(lambda x: kernel_size_check(x, kernel_size))


def test_encoder_2d_contents_with_different_max_feats():
    """Test the contents of ``Encoder2d`` with different ``max_feats``."""
    # pylint: disable=cell-var-from-loop

    def down_max_feats(model: Module, max_feats: int):
        """Check the max feats is correct."""

        if isinstance(model, DownBlock) and (max_feats is not None):
            assert model[1][0][0].in_channels <= max_feats
            assert model[1][0][0].out_channels <= max_feats
            assert model[1][0][1].num_features <= max_feats

            assert model[1][1][0].in_channels <= max_feats
            assert model[1][1][0].out_channels <= max_feats
            assert model[1][1][1].num_features <= max_feats

            print(model)

    for max_feats in [None, 1, 128, 256]:
        encoder = Encoder2d(
            in_chans=123,
            start_features=64,
            num_blocks=5,
            pool_style="max",
            lr_slope=0.666,
            kernel_size=3,
            max_feats=max_feats,
        )

        encoder.apply(lambda x: down_max_feats(x, max_feats))


def test_encoder_2d_contents_with_double_conv_blocks():
    """Test the contents of ``Encoder2d`` with different block styles."""
    encoder = Encoder2d(
        in_chans=123,
        start_features=64,
        num_blocks=5,
        pool_style="max",
        lr_slope=0.666,
        kernel_size=3,
        block_style="double_conv",
    )

    in_chans, out_chans = 64, 128

    for block in list(encoder.children())[1:]:
        assert isinstance(block[1], DoubleConvBlock)

        # Test the first conv block of the double conv
        assert isinstance(block[1][0], ConvBlock)
        assert isinstance(block[1][0][0], Conv2d)
        assert isinstance(block[1][0][1], BatchNorm2d)
        assert isinstance(block[1][0][2], LeakyReLU)

        assert block[1][0][0].in_channels == in_chans
        assert block[1][0][0].out_channels == out_chans
        assert block[1][0][1].num_features == out_chans

        # Test the second conv block of the double conv
        assert isinstance(block[1][1], ConvBlock)
        assert isinstance(block[1][1][0], Conv2d)
        assert isinstance(block[1][1][1], BatchNorm2d)
        assert isinstance(block[1][1][2], LeakyReLU)

        assert block[1][1][0].in_channels == out_chans
        assert block[1][1][0].out_channels == out_chans
        assert block[1][1][1].num_features == out_chans

        in_chans *= 2
        out_chans *= 2


def test_encoder_2d_contents_with_conv_res_blocks():
    """Test the contents of ``Encoder2d`` with different block styles."""
    encoder = Encoder2d(
        in_chans=123,
        start_features=64,
        num_blocks=5,
        pool_style="max",
        lr_slope=0.666,
        kernel_size=3,
        block_style="conv_res",
    )

    in_chans, out_chans = 64, 128

    for block in list(encoder.children())[1:]:
        assert isinstance(block[1], ConvResBlock)

        # Test the first conv block of the conv res block
        assert isinstance(block[1][0], ConvBlock)
        assert isinstance(block[1][0][0], Conv2d)
        assert isinstance(block[1][0][1], BatchNorm2d)
        assert isinstance(block[1][0][2], LeakyReLU)

        assert block[1][0][0].in_channels == in_chans
        assert block[1][0][0].out_channels == out_chans
        assert block[1][0][1].num_features == out_chans

        # Test the residual block
        assert isinstance(block[1][1], ResidualBlock)
        assert isinstance(block[1][1].first_conv, ConvBlock)
        assert isinstance(block[1][1].first_conv[0], Conv2d)
        assert isinstance(block[1][1].first_conv[1], BatchNorm2d)
        assert isinstance(block[1][1].first_conv[2], LeakyReLU)

        assert block[1][1].first_conv[0].in_channels == out_chans
        assert block[1][1].first_conv[0].out_channels == out_chans
        assert block[1][1].first_conv[1].num_features == out_chans

        assert isinstance(block[1][1], ResidualBlock)
        assert isinstance(block[1][1].second_conv, ConvBlock)
        assert isinstance(block[1][1].second_conv[0], Conv2d)
        assert isinstance(block[1][1].second_conv[1], BatchNorm2d)

        assert block[1][1].second_conv[0].in_channels == out_chans
        assert block[1][1].second_conv[0].out_channels == out_chans
        assert block[1][1].second_conv[1].num_features == out_chans

        in_chans *= 2
        out_chans *= 2
