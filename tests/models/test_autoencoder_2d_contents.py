"""Test the contents of ``torch_tools.AutoEncoder2d``."""
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, AvgPool2d, MaxPool2d
from torch.nn import ConvTranspose2d, Sequential, Upsample

from torch_tools import AutoEncoder2d
from torch_tools.models._blocks_2d import DoubleConvBlock, ConvBlock, DownBlock
from torch_tools.models._blocks_2d import UpBlock


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
