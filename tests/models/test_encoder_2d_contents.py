"""Test the contents of `torch_tools.models._encoder_2d.Encoder2d`."""

from torch.nn import Conv2d, BatchNorm2d, LeakyReLU

from torch_tools import Encoder2d

from torch_tools.models._blocks_2d import DoubleConvBlock, ConvBlock


def test_encoder_2d_total_number_of_blocks():
    """Test the number of blocks in the encoder."""
    encoder = Encoder2d(
        in_chans=123,
        start_features=64,
        num_blocks=4,
        pool_style="avg",
        lr_slope=0.123,
    )
    assert len(encoder) == 4, "Wrong number of blocks in the encoder."

    encoder = Encoder2d(
        in_chans=123,
        start_features=64,
        num_blocks=7,
        pool_style="avg",
        lr_slope=0.123,
    )
    assert len(encoder) == 7, "Wrong number of blocks in the encoder."


def test_encoder_2d_double_conv_contents():
    """Test the contents of the double conv layer."""
    encoder = Encoder2d(
        in_chans=123,
        start_features=64,
        num_blocks=4,
        pool_style="avg",
        lr_slope=0.123,
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
    assert first_conv_block[0].out_channels == 64
    assert first_conv_block[1].num_features == 64
    assert first_conv_block[2].negative_slope == 0.123

    # Test the second conv block
    assert isinstance(second_conv_block, ConvBlock)
    assert isinstance(second_conv_block[0], Conv2d)
    assert isinstance(second_conv_block[1], BatchNorm2d)
    assert isinstance(second_conv_block[2], LeakyReLU)

    assert second_conv_block[0].in_channels == 64
    assert second_conv_block[0].out_channels == 64
    assert second_conv_block[1].num_features == 64
    assert second_conv_block[2].negative_slope == 0.123
