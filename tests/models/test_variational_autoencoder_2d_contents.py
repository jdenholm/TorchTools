"""Test the contents of ``VAE2d``."""
from torch_tools import VAE2d


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
