"""Test the contents of ``VAE2d``."""

from itertools import product

from torch.nn import Module, Conv2d, LeakyReLU, BatchNorm2d, Sequential
from torch.nn import Dropout2d

from torch_tools import VAE2d, FCNet
from torch_tools.models._blocks_2d import DownBlock, ConvBlock, ConvResBlock
from torch_tools.models._blocks_2d import ResidualBlock, DoubleConvBlock
from torch_tools.models._blocks_2d import UpBlock


def test_number_of_blocks_in_encoder_and_decoder():
    """Test the number of blocks in the encoder and decoder."""
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(64, 64),
        num_layers=5,
        start_features=8,
    )
    assert len(model.encoder) == 5
    assert len(model.decoder) == 5

    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(64, 64),
        num_layers=3,
        start_features=8,
    )
    assert len(model.encoder) == 3
    assert len(model.decoder) == 3


def test_mean_net_in_feats():
    """Test the number of features in the mean network."""
    # pylint: disable=protected-access
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(32, 32),
        num_layers=4,
        start_features=8,
    )

    assert model.mean_net[0][0].in_features == 1024


def test_std_net_in_feats():
    """Test the number of features in the std network."""
    # pylint: disable=protected-access
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(32, 32),
        num_layers=3,
        start_features=8,
    )

    assert model.var_net[0][0].in_features == 2048


def test_mean_net_out_feats():
    """Test the number of features in the mean network."""
    # pylint: disable=protected-access
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(64, 64),
        num_layers=6,
        start_features=8,
    )

    assert model.mean_net[0][0].out_features == 1024


def test_mean_net_out_feats_with_max_down_feats():
    """Test the number of features in the mean network."""
    # pylint: disable=protected-access
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(64, 64),
        num_layers=7,
        start_features=64,
        max_down_feats=64,
    )

    assert model.mean_net[0][0].out_features == 64


def test_std_net_out_feats():
    """Test the number of features in the std network."""
    # pylint: disable=protected-access
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(32, 32),
        num_layers=5,
        start_features=8,
    )

    assert model.var_net[0][0].out_features == 512


def test_std_net_out_feats_with_max_down_feats():
    """Test the number of features in the std network."""
    # pylint: disable=protected-access
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(128, 128),
        num_layers=8,
        start_features=64,
        max_down_feats=128,
    )

    assert model.var_net[0][0].out_features == 128


def test_vae_2d_contents_with_different_max_feats():
    """Test the contents of the encoder with different ``max_feats``."""
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

    for max_feats in [None, 32, 64]:
        model = VAE2d(
            in_chans=3,
            out_chans=3,
            start_features=16,
            input_dims=(64, 64),
            num_layers=5,
            lr_slope=0.666,
            kernel_size=3,
            max_down_feats=max_feats,
        )

        model.encoder.apply(lambda x: down_max_feats(x, max_feats))


def test_vae_2d_contents_with_different_min_up_feats():
    """Test the contents of the decoder with different ``min_up_feats`` arg."""
    # pylint: disable=cell-var-from-loop

    def down_max_feats(model: Module, min_feats: int):
        """Check the min feats is correct."""

        if isinstance(model, DownBlock) and (min_feats is not None):
            assert model[1][0][0].in_channels <= min_feats
            assert model[1][0][0].out_channels <= min_feats
            assert model[1][0][1].num_features <= min_feats

            assert model[1][1][0].in_channels <= min_feats
            assert model[1][1][0].out_channels <= min_feats
            assert model[1][1][1].num_features <= min_feats

            print(model)

    for min_feats in [None, 8, 16]:
        model = VAE2d(
            in_chans=3,
            out_chans=3,
            start_features=32,
            input_dims=(64, 64),
            num_layers=5,
            lr_slope=0.666,
            kernel_size=3,
            min_up_feats=min_feats,
        )

        model.decoder.apply(lambda x: down_max_feats(x, min_feats))


def test_vae_2d_encoder_contents_with_conv_res_blocks():
    """Test the contents of the encoder in ``VAE2d`` with conv res blocks."""
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        start_features=8,
        input_dims=(32, 32),
        block_style="conv_res",
        lr_slope=0.123,
    )

    encoder = model.encoder

    in_chans, out_chans = 8, 16

    for block in list(encoder.children())[1:]:
        assert isinstance(block, DownBlock)
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


def test_vae_2d_encoder_contents_with_double_conv_blocks():
    """Test the contents of the encoder in ``VAE2d`` with conv res blocks."""
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        start_features=16,
        input_dims=(32, 32),
        block_style="double_conv",
        lr_slope=0.123,
    )

    encoder = model.encoder

    in_chans, out_chans = 16, 32

    for block in list(encoder.children())[1:]:
        assert isinstance(block, DownBlock)
        assert isinstance(block[1], DoubleConvBlock)
        assert isinstance(block[1][0], ConvBlock)

        assert isinstance(block[1][0][0], Conv2d)
        assert isinstance(block[1][0][1], BatchNorm2d)
        assert isinstance(block[1][0][2], LeakyReLU)

        assert block[1][0][0].in_channels == in_chans
        assert block[1][0][0].out_channels == out_chans
        assert block[1][0][1].num_features == out_chans
        assert block[1][0][2].negative_slope == 0.123

        assert isinstance(block[1][1][0], Conv2d)
        assert isinstance(block[1][1][1], BatchNorm2d)
        assert isinstance(block[1][1][2], LeakyReLU)

        assert block[1][1][0].in_channels == out_chans
        assert block[1][1][0].out_channels == out_chans
        assert block[1][1][1].num_features == out_chans
        assert block[1][1][2].negative_slope == 0.123

        in_chans *= 2
        out_chans *= 2


def test_vae_2d_decoder_contents_with_double_conv_blocks():
    """Test the contents of the decoder in ``VAE2d`` with conv res blocks."""
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        start_features=8,
        input_dims=(32, 32),
        block_style="double_conv",
        lr_slope=0.123,
        num_layers=3,
    )

    decoder = model.decoder

    in_chans, out_chans = 8 * (2**2), 8 * (2**1)

    for block in list(decoder.children())[:-1]:
        assert isinstance(block, UpBlock)
        assert isinstance(block[1], DoubleConvBlock)
        assert isinstance(block[1][0], ConvBlock)

        assert isinstance(block[1][0][0], Conv2d)
        assert isinstance(block[1][0][1], BatchNorm2d)
        assert isinstance(block[1][0][2], LeakyReLU)

        assert block[1][0][0].in_channels == in_chans
        assert block[1][0][0].out_channels == out_chans
        assert block[1][0][1].num_features == out_chans
        assert block[1][0][2].negative_slope == 0.123

        assert isinstance(block[1][1][0], Conv2d)
        assert isinstance(block[1][1][1], BatchNorm2d)
        assert isinstance(block[1][1][2], LeakyReLU)

        assert block[1][1][0].in_channels == out_chans
        assert block[1][1][0].out_channels == out_chans
        assert block[1][1][1].num_features == out_chans
        assert block[1][1][2].negative_slope == 0.123

        in_chans //= 2
        out_chans //= 2


def test_vae_2d_decoder_contents_with_conv_res_blocks():
    """Test the contents of the decoder in ``VAE2d`` with conv res blocks."""
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        start_features=8,
        input_dims=(32, 32),
        block_style="conv_res",
        lr_slope=0.123,
        num_layers=3,
    )

    decoder = model.decoder

    in_chans, out_chans = 8 * (2**2), 8 * (2**1)

    for block in list(decoder.children())[:-1]:
        assert isinstance(block, UpBlock)
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

        in_chans //= 2
        out_chans //= 2


def test_vae_contents_with_linear_mean_var():
    """Test the contents of the model with linear mean and var nets."""
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        start_features=8,
        input_dims=(64, 64),
        block_style="conv_res",
        lr_slope=0.123,
        num_layers=3,
        mean_var_nets="linear",
    )

    assert isinstance(model.mean_net, FCNet)


def test_vae_contents_with_conv_mean_var():
    """Test the contents of the model with linear mean and var nets."""
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        start_features=8,
        input_dims=None,
        block_style="conv_res",
        lr_slope=0.123,
        num_layers=3,
        mean_var_nets="conv",
    )

    assert isinstance(model.mean_net, Sequential)
    assert isinstance(model.mean_net[0], DoubleConvBlock)
    assert isinstance(model.mean_net[1], Conv2d)


def test_vae_encoder_contents_with_dropout():
    """Test the contents of the encoder with dropout."""
    blocks = {"double_conv": DoubleConvBlock, "conv_res": ConvResBlock}

    for block_style, dropout in product(blocks.keys(), [0.0, 0.666]):

        encoder = VAE2d(
            in_chans=3,
            out_chans=3,
            start_features=8,
            input_dims=(64, 64),
            block_style=block_style,
            lr_slope=0.123,
            num_layers=3,
            mean_var_nets="linear",
            dropout=dropout,
        ).encoder

        # Test the first conv block
        assert isinstance(encoder[0], blocks[block_style])
        assert len(encoder[0]) == 3 if dropout != 0.0 else 2
        if dropout != 0.0:
            assert isinstance(encoder[0][2], Dropout2d)
            assert encoder[0][2].p == dropout

        # Test the down blocks
        for block in list(encoder.children())[1:]:

            assert isinstance(block, DownBlock)
            assert isinstance(block[1], blocks[block_style])
            assert len(block[1]) == 3 if dropout != 0.0 else 2
            if dropout != 0.0:
                assert isinstance(block[1][2], Dropout2d)
                assert block[1][2].p == dropout


def test_vae_decoder_contents_with_dropout():
    """Test the contents of the decoder with dropout."""
    blocks = {"double_conv": DoubleConvBlock, "conv_res": ConvResBlock}

    for block_style, dropout in product(blocks.keys(), [0.0, 0.666]):

        decoder = VAE2d(
            in_chans=3,
            out_chans=3,
            start_features=8,
            input_dims=(64, 64),
            block_style=block_style,
            lr_slope=0.123,
            num_layers=3,
            mean_var_nets="linear",
            dropout=dropout,
        ).decoder

        for block in list(decoder.children())[:-1]:

            assert isinstance(block, UpBlock)
            assert isinstance(block[1], blocks[block_style])

            assert len(block[1]) == 3 if dropout != 0.0 else 2
            if dropout != 0.0:
                assert isinstance(block[1][2], Dropout2d)
                assert block[1][2].p == dropout
