"""Test the contents of ``VAE2d``."""
from torch.nn import Module, Conv2d, LeakyReLU, BatchNorm2d

from torch_tools import VAE2d
from torch_tools.models._blocks_2d import DownBlock, ConvBlock, ConvResBlock
from torch_tools.models._blocks_2d import ResidualBlock


def test_number_of_blocks_in_encoder_and_decoder():
    """Test the number of blocks in the encoder and decoder."""
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(256, 256),
        num_layers=7,
        start_features=8,
    )
    assert len(model.encoder) == 7
    assert len(model.decoder) == 7

    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(512, 512),
        num_layers=8,
        start_features=8,
    )
    assert len(model.encoder) == 8
    assert len(model.decoder) == 8


def test_mean_net_in_feats():
    """Test the number of features in the mean network."""
    # pylint: disable=protected-access
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(256, 256),
        num_layers=7,
        start_features=8,
    )

    assert model._mean_net[0][0].in_features == 8192


def test_std_net_in_feats():
    """Test the number of features in the std network."""
    # pylint: disable=protected-access
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(256, 256),
        num_layers=7,
        start_features=8,
    )

    assert model._var_net[0][0].in_features == 8192


def test_mean_net_out_feats():
    """Test the number of features in the mean network."""
    # pylint: disable=protected-access
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(256, 256),
        num_layers=7,
        start_features=8,
    )

    assert model._mean_net[0][0].out_features == 8192


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

    assert model._mean_net[0][0].out_features == 64


def test_std_net_out_feats():
    """Test the number of features in the std network."""
    # pylint: disable=protected-access
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        input_dims=(256, 256),
        num_layers=7,
        start_features=8,
    )

    assert model._var_net[0][0].out_features == 8192


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

    assert model._var_net[0][0].out_features == 128


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

    for max_feats in [None, 128, 256]:
        model = VAE2d(
            in_chans=3,
            out_chans=3,
            start_features=64,
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
            start_features=64,
            input_dims=(64, 64),
            num_layers=5,
            lr_slope=0.666,
            kernel_size=3,
            min_up_feats=min_feats,
        )

        model.decoder.apply(lambda x: down_max_feats(x, min_feats))


def test_vae_2d_encoder_contents_with_conv_res_blocks():
    """Test the contents of the encoder in ``VAE2d``."""
    model = VAE2d(
        in_chans=3,
        out_chans=3,
        start_features=32,
        input_dims=(64, 64),
        block_style="conv_res",
        lr_slope=0.123,
    )

    encoder = model.encoder

    in_chans, out_chans = 32, 64

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
