"""Test the contents of ``torch_tools.AutoEncoder2d``."""
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, AvgPool2d, MaxPool2d
from torch.nn import ConvTranspose2d, Sequential, Upsample, Module

from torch_tools import AutoEncoder2d
from torch_tools.models._blocks_2d import DoubleConvBlock, ConvBlock, DownBlock
from torch_tools.models._blocks_2d import UpBlock, ConvResBlock, ResidualBlock


def kernel_size_check(block: Module, kernel_size: int):
    """Make sure any kernel sizes are correct."""
    if isinstance(block, Conv2d):
        assert block.kernel_size == (kernel_size, kernel_size)


def test_number_of_blocks():
    """Test the number of blocks in the encoder and decoder."""
    model = AutoEncoder2d(in_chans=3, out_chans=3, num_layers=5)
    assert len(model.encoder) == 5
    assert len(model.decoder) == 5

    model = AutoEncoder2d(in_chans=3, out_chans=3, num_layers=3)
    assert len(model.encoder) == 3
    assert len(model.decoder) == 3


def test_encoder_double_conv_contents():
    """Test the contents of the double conv block in the encoder."""
    model = AutoEncoder2d(in_chans=123, out_chans=321, features_start=64)
    encoder = model.encoder

    assert isinstance(encoder[0], DoubleConvBlock)

    first_conv, second_conv = encoder[0]
    assert isinstance(first_conv, ConvBlock)
    assert isinstance(first_conv[0], Conv2d)
    assert isinstance(first_conv[1], BatchNorm2d)
    assert isinstance(first_conv[2], LeakyReLU)

    assert first_conv[0].in_channels == 123
    assert first_conv[0].out_channels == 64
    assert first_conv[1].num_features == 64

    assert isinstance(second_conv, ConvBlock)
    assert isinstance(second_conv[0], Conv2d)
    assert isinstance(second_conv[1], BatchNorm2d)
    assert isinstance(second_conv[2], LeakyReLU)

    assert second_conv[0].in_channels == 64
    assert second_conv[0].out_channels == 64
    assert second_conv[1].num_features == 64


def test_encoder_down_block_contents_with_avg_pool():
    """Test the contents of the down blocks in the encoder with avg pools."""
    model = AutoEncoder2d(
        in_chans=123,
        out_chans=321,
        features_start=64,
        pool_style="avg",
    )
    encoder = model.encoder

    in_chans = 64
    for down_block in list(encoder.children())[1:]:
        assert isinstance(down_block, DownBlock)
        assert isinstance(down_block[0], AvgPool2d)

        assert isinstance(down_block[1], DoubleConvBlock)
        assert isinstance(down_block[1][1], ConvBlock)

        first_conv, second_conv = down_block[1]

        assert isinstance(first_conv[0], Conv2d)
        assert isinstance(first_conv[1], BatchNorm2d)
        assert isinstance(first_conv[2], LeakyReLU)

        assert isinstance(second_conv[0], Conv2d)
        assert isinstance(second_conv[1], BatchNorm2d)
        assert isinstance(second_conv[2], LeakyReLU)

        assert first_conv[0].in_channels == in_chans
        assert first_conv[0].out_channels == in_chans * 2
        assert first_conv[1].num_features == in_chans * 2

        assert second_conv[0].in_channels == in_chans * 2
        assert second_conv[0].out_channels == in_chans * 2
        assert second_conv[1].num_features == in_chans * 2

        in_chans *= 2


def test_encoder_down_block_contents_with_max_pool():
    """Test the contents of the down blocks in the encoder with avg pools."""
    model = AutoEncoder2d(
        in_chans=123,
        out_chans=321,
        features_start=64,
        pool_style="max",
    )
    encoder = model.encoder

    in_chans = 64
    for down_block in list(encoder.children())[1:]:
        assert isinstance(down_block, DownBlock)
        assert isinstance(down_block[0], MaxPool2d)

        assert isinstance(down_block[1], DoubleConvBlock)
        assert isinstance(down_block[1][1], ConvBlock)

        first_conv, second_conv = down_block[1]

        assert isinstance(first_conv[0], Conv2d)
        assert isinstance(first_conv[1], BatchNorm2d)
        assert isinstance(first_conv[2], LeakyReLU)

        assert isinstance(second_conv[0], Conv2d)
        assert isinstance(second_conv[1], BatchNorm2d)
        assert isinstance(second_conv[2], LeakyReLU)

        assert first_conv[0].in_channels == in_chans
        assert first_conv[0].out_channels == in_chans * 2
        assert first_conv[1].num_features == in_chans * 2

        assert second_conv[0].in_channels == in_chans * 2
        assert second_conv[0].out_channels == in_chans * 2
        assert second_conv[1].num_features == in_chans * 2

        in_chans *= 2


def test_deconder_final_conv():
    """Test the final layer in the decoder is a conv 2d."""
    model = AutoEncoder2d(
        in_chans=123,
        out_chans=321,
        features_start=64,
        pool_style="avg",
    )

    decoder = model.decoder

    assert isinstance(decoder[-1], Conv2d)
    assert decoder[-1].in_channels == 64
    assert decoder[-1].out_channels == 321


def test_decoder_up_block_contents_with_conv_transpose():
    """Test the contents of the decoder with conv transpose upsampling."""
    model = AutoEncoder2d(
        in_chans=123,
        out_chans=321,
        features_start=64,
        pool_style="avg",
        bilinear=False,
        num_layers=4,
    )

    decoder = model.decoder

    in_chans = 64 * (2**3)
    for up_block in list(decoder.children())[:-1]:
        assert isinstance(up_block, UpBlock)
        assert isinstance(up_block[0], ConvTranspose2d)

        assert isinstance(up_block[1], DoubleConvBlock)

        first_conv, second_conv = up_block[1]

        assert isinstance(first_conv, ConvBlock)
        assert isinstance(first_conv[0], Conv2d)
        assert isinstance(first_conv[1], BatchNorm2d)
        assert isinstance(first_conv[2], LeakyReLU)

        assert isinstance(second_conv, ConvBlock)
        assert isinstance(second_conv[0], Conv2d)
        assert isinstance(second_conv[1], BatchNorm2d)
        assert isinstance(second_conv[2], LeakyReLU)

        assert first_conv[0].in_channels == in_chans
        assert first_conv[0].out_channels == in_chans // 2
        assert first_conv[1].num_features == in_chans // 2

        assert second_conv[0].in_channels == in_chans // 2
        assert second_conv[0].out_channels == in_chans // 2
        assert second_conv[1].num_features == in_chans // 2

        in_chans //= 2


def test_decoder_up_block_contents_with_bilinear_interpolation():
    """Test contents of the decoder with bilinear interpolation upsampling."""
    model = AutoEncoder2d(
        in_chans=123,
        out_chans=321,
        features_start=64,
        pool_style="avg",
        bilinear=True,
        num_layers=4,
    )

    decoder = model.decoder

    in_chans = 64 * (2**3)
    for up_block in list(decoder.children())[:-1]:
        assert isinstance(up_block, UpBlock)

        assert isinstance(up_block[0], Sequential)
        assert isinstance(up_block[0][0], Upsample)
        assert up_block[0][0].mode == "bilinear"
        assert isinstance(up_block[0][1], Conv2d)

        assert isinstance(up_block[1], DoubleConvBlock)

        first_conv, second_conv = up_block[1]

        assert isinstance(first_conv, ConvBlock)
        assert isinstance(first_conv[0], Conv2d)
        assert isinstance(first_conv[1], BatchNorm2d)
        assert isinstance(first_conv[2], LeakyReLU)

        assert isinstance(second_conv, ConvBlock)
        assert isinstance(second_conv[0], Conv2d)
        assert isinstance(second_conv[1], BatchNorm2d)
        assert isinstance(second_conv[2], LeakyReLU)

        assert first_conv[0].in_channels == in_chans
        assert first_conv[0].out_channels == in_chans // 2
        assert first_conv[1].num_features == in_chans // 2

        assert second_conv[0].in_channels == in_chans // 2
        assert second_conv[0].out_channels == in_chans // 2
        assert second_conv[1].num_features == in_chans // 2

        in_chans //= 2


def test_autoencoder_encoder_contents_with_different_kernel_sizes():
    """Test the ``AutoEncoder2d``s encoder's contents."""
    for size in [1, 3, 5]:
        auto_encoder = AutoEncoder2d(in_chans=3, out_chans=3, kernel_size=size)

        encoder = auto_encoder.encoder
        encoder.apply(lambda x: kernel_size_check(x, size))


def test_autoencoder_decoder_contents_with_different_kernel_sizes():
    """Test the ``AutoEncoder2d``'s decoder's contents."""
    for size in [1, 3, 5]:
        auto_encoder = AutoEncoder2d(in_chans=3, out_chans=3, kernel_size=size)
        decoder = auto_encoder.decoder

        for block in decoder.children():
            # The only conv blocks whose kernel size should change
            # are in the UpBlocks
            if isinstance(block, UpBlock):
                block[1].apply(lambda x: kernel_size_check(x, size))


def test_encoder_contents_with_double_conv_block():
    """Test the contents of the encoder with double conv blocks."""
    model = AutoEncoder2d(
        in_chans=3,
        out_chans=3,
        features_start=64,
        block_style="double_conv",
        lr_slope=0.123,
    )

    in_chans, out_chans = 64, 128

    for block in list(model.encoder.children())[1:]:
        assert isinstance(block[1], DoubleConvBlock)
        assert isinstance(block[1][0], ConvBlock)

        # Test the first conv block
        assert isinstance(block[1][0][0], Conv2d)
        assert isinstance(block[1][0][1], BatchNorm2d)
        assert isinstance(block[1][0][2], LeakyReLU)

        assert block[1][0][0].in_channels == in_chans
        assert block[1][0][0].out_channels == out_chans
        assert block[1][0][1].num_features == out_chans
        assert block[1][0][2].negative_slope == 0.123

        # Test the second conv block
        assert isinstance(block[1][1][0], Conv2d)
        assert isinstance(block[1][1][1], BatchNorm2d)
        assert isinstance(block[1][1][2], LeakyReLU)

        assert block[1][1][0].in_channels == out_chans
        assert block[1][1][0].out_channels == out_chans
        assert block[1][1][1].num_features == out_chans
        assert block[1][1][2].negative_slope == 0.123

        in_chans *= 2
        out_chans *= 2


def test_decoder_contents_with_double_conv_block():
    """Test the contents of the decoder with double conv blocks."""
    model = AutoEncoder2d(
        in_chans=3,
        out_chans=3,
        features_start=64,
        block_style="double_conv",
        lr_slope=0.123,
        num_layers=4,
    )

    in_chans = 64 * 2 ** (3)
    out_chans = in_chans // 2

    for block in list(model.decoder.children())[:-1]:
        assert isinstance(block[1], DoubleConvBlock)
        assert isinstance(block[1][0], ConvBlock)

        # Test the first conv block
        assert isinstance(block[1][0][0], Conv2d)
        assert isinstance(block[1][0][1], BatchNorm2d)
        assert isinstance(block[1][0][2], LeakyReLU)

        assert block[1][0][0].in_channels == in_chans
        assert block[1][0][0].out_channels == out_chans
        assert block[1][0][1].num_features == out_chans
        assert block[1][0][2].negative_slope == 0.123

        # Test the second conv block
        assert isinstance(block[1][1][0], Conv2d)
        assert isinstance(block[1][1][1], BatchNorm2d)
        assert isinstance(block[1][1][2], LeakyReLU)

        assert block[1][1][0].in_channels == out_chans
        assert block[1][1][0].out_channels == out_chans
        assert block[1][1][1].num_features == out_chans
        assert block[1][1][2].negative_slope == 0.123

        in_chans /= 2
        out_chans /= 2


def test_encoder_contents_with_conv_res_block():
    """Test the contents of the encoder with conv res blocks."""
    model = AutoEncoder2d(
        in_chans=3,
        out_chans=3,
        features_start=64,
        block_style="conv_res",
        lr_slope=0.123,
    )

    in_chans, out_chans = 64, 128

    for block in list(model.encoder.children())[1:]:
        assert isinstance(block[1], ConvResBlock)
        assert isinstance(block[1][0], ConvBlock)
        assert isinstance(block[1][0][0], Conv2d)
        assert isinstance(block[1][0][1], BatchNorm2d)
        assert isinstance(block[1][0][2], LeakyReLU)

        assert block[1][0][0].in_channels == in_chans
        assert block[1][0][0].out_channels == out_chans
        assert block[1][0][1].num_features == out_chans
        assert block[1][0][2].negative_slope == 0.123

        assert isinstance(block[1][1], ResidualBlock)
        assert isinstance(block[1][1].first_conv, ConvBlock)
        assert isinstance(block[1][1].first_conv[0], Conv2d)
        assert isinstance(block[1][1].first_conv[1], BatchNorm2d)
        assert isinstance(block[1][1].first_conv[2], LeakyReLU)

        assert isinstance(block[1][1], ResidualBlock)
        assert isinstance(block[1][1].second_conv, ConvBlock)
        assert isinstance(block[1][1].second_conv[0], Conv2d)
        assert isinstance(block[1][1].second_conv[1], BatchNorm2d)

        assert block[1][1].first_conv[0].in_channels == out_chans
        assert block[1][1].first_conv[0].out_channels == out_chans
        assert block[1][1].first_conv[1].num_features == out_chans
        assert block[1][1].first_conv[2].negative_slope == 0.0

        assert block[1][1].second_conv[0].in_channels == out_chans
        assert block[1][1].second_conv[0].out_channels == out_chans
        assert block[1][1].second_conv[1].num_features == out_chans

        in_chans *= 2
        out_chans *= 2


def test_decoder_contents_with_conv_res_block():
    """Test the contents of the decoder with conv res blocks."""
    model = AutoEncoder2d(
        in_chans=3,
        out_chans=3,
        features_start=64,
        block_style="conv_res",
        lr_slope=0.123,
        num_layers=4,
    )

    in_chans = 64 * 2 ** (3)
    out_chans = in_chans // 2

    for block in list(model.decoder.children())[:-1]:
        assert isinstance(block[1], ConvResBlock)
        assert isinstance(block[1][0], ConvBlock)
        assert isinstance(block[1][0][0], Conv2d)
        assert isinstance(block[1][0][1], BatchNorm2d)
        assert isinstance(block[1][0][2], LeakyReLU)

        assert block[1][0][0].in_channels == in_chans
        assert block[1][0][0].out_channels == out_chans
        assert block[1][0][1].num_features == out_chans
        assert block[1][0][2].negative_slope == 0.123

        assert isinstance(block[1][1], ResidualBlock)
        assert isinstance(block[1][1].first_conv, ConvBlock)
        assert isinstance(block[1][1].first_conv[0], Conv2d)
        assert isinstance(block[1][1].first_conv[1], BatchNorm2d)
        assert isinstance(block[1][1].first_conv[2], LeakyReLU)

        assert isinstance(block[1][1], ResidualBlock)
        assert isinstance(block[1][1].second_conv, ConvBlock)
        assert isinstance(block[1][1].second_conv[0], Conv2d)
        assert isinstance(block[1][1].second_conv[1], BatchNorm2d)

        assert block[1][1].first_conv[0].in_channels == out_chans
        assert block[1][1].first_conv[0].out_channels == out_chans
        assert block[1][1].first_conv[1].num_features == out_chans
        assert block[1][1].first_conv[2].negative_slope == 0.0

        assert block[1][1].second_conv[0].in_channels == out_chans
        assert block[1][1].second_conv[0].out_channels == out_chans
        assert block[1][1].second_conv[1].num_features == out_chans

        in_chans //= 2
        out_chans //= 2
