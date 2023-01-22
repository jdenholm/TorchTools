"""Tests for the contents of `torch_tools.models._decoder_2d.Decoder_2d`."""

from torch.nn import Conv2d, ConvTranspose2d, Sequential, Upsample
from torch.nn import BatchNorm2d, LeakyReLU

from torch_tools import Decoder2d
from torch_tools.models._blocks_2d import UpBlock, DoubleConvBlock, ConvBlock


def test_decoder_2d_number_of_blocks():
    """Test the number of blocks in `Decoder2d`."""
    # Try with 5 blocks
    decoder = Decoder2d(
        in_chans=512,
        out_chans=1,
        num_blocks=5,
        bilinear=False,
        lr_slope=0.123456,
    )

    assert len(decoder) == 5

    # Try with 3 blocks
    decoder = Decoder2d(
        in_chans=256,
        out_chans=1,
        num_blocks=3,
        bilinear=False,
        lr_slope=0.123456,
    )

    assert len(decoder) == 3


def test_decoder_2d_block_types():
    """Test the blocks are the correct types."""
    decoder = Decoder2d(
        in_chans=512,
        out_chans=1,
        num_blocks=5,
        bilinear=False,
        lr_slope=0.123456,
    )

    # All but the last block should be down blocks
    down_blocks = list(decoder.children())[:-1]
    assert all(map(lambda x: isinstance(x, UpBlock), down_blocks))

    # The last block should be a simple Conv2d
    final_conv = list(decoder.children())[-1]
    assert isinstance(final_conv, Conv2d)


def test_decoder_2d_final_layer():
    """Test the final layer is of the correct types with correct attributes."""
    decoder = Decoder2d(
        in_chans=512,
        out_chans=123,
        num_blocks=5,
        bilinear=False,
        lr_slope=0.123456,
    )

    # The last layer should be a conv 2d
    assert isinstance(decoder[-1], Conv2d)
    assert decoder[-1].in_channels == 512 // (2**4)
    assert decoder[-1].out_channels == 123

    assert decoder[-1].kernel_size == (1, 1)
    assert decoder[-1].stride == (1, 1)


def test_decoder_2d_up_layer_types_with_bilinear_false():
    """Make sure the model contains the correct upsampling layers."""
    decoder = Decoder2d(
        in_chans=512,
        out_chans=123,
        num_blocks=5,
        bilinear=False,
        lr_slope=0.123456,
    )

    for block in list(decoder.children())[:-1]:

        assert isinstance(block[0], ConvTranspose2d)


def test_decoder_2d_up_layer_types_with_bilinear_true():
    """Make sure the model contains the correct upsampling layers."""
    decoder = Decoder2d(
        in_chans=512,
        out_chans=123,
        num_blocks=5,
        bilinear=True,
        lr_slope=0.123456,
    )

    for block in list(decoder.children())[:-1]:

        assert isinstance(block[0], Sequential)
        assert isinstance(block[0][0], Upsample)
        assert isinstance(block[0][1], Conv2d)


def test_first_up_block_contents_with_bilinear_false():
    """Test the contents of the first up block."""
    decoder = Decoder2d(
        in_chans=512,
        out_chans=123,
        num_blocks=5,
        bilinear=False,
        lr_slope=0.123456,
    )

    first_block = decoder[0]

    assert isinstance(first_block, UpBlock)
    assert isinstance(first_block[0], ConvTranspose2d)
    assert isinstance(first_block[1], DoubleConvBlock)

    # The double conv block should be made up of two single conv blocks

    # Test the first conv block
    assert isinstance(first_block[1][0], ConvBlock)
    assert isinstance(first_block[1][0][0], Conv2d)
    assert isinstance(first_block[1][0][1], BatchNorm2d)
    assert isinstance(first_block[1][0][2], LeakyReLU)

    assert first_block[1][0][0].in_channels == 512
    assert first_block[1][0][0].out_channels == 256
    assert first_block[1][0][1].num_features == 256
    assert first_block[1][0][2].negative_slope == 0.123456

    # Test the second conv block
    assert isinstance(first_block[1][1], ConvBlock)
    assert isinstance(first_block[1][1][0], Conv2d)
    assert isinstance(first_block[1][1][1], BatchNorm2d)
    assert isinstance(first_block[1][1][2], LeakyReLU)

    assert first_block[1][1][0].in_channels == 256
    assert first_block[1][1][0].out_channels == 256
    assert first_block[1][1][1].num_features == 256
    assert first_block[1][1][2].negative_slope == 0.123456
