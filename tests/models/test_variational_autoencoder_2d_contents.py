"""Test the contents of ``VAE2d``."""
from torch.nn import Module

from torch_tools import VAE2d
from torch_tools.models._blocks_2d import DownBlock


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

    assert model._std_net[0][0].in_features == 8192


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

    assert model._std_net[0][0].out_features == 8192


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

    assert model._std_net[0][0].out_features == 128


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
