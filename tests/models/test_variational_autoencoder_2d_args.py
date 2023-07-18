"""Test for the arguments of ``VAE2d``."""
import pytest

from torch_tools.models._variational_autoencoder_2d import VAE2d


def test_in_chans_arg_types():
    """Test the types accepted by the ``in_chans`` argument."""
    # Should work with positive int
    _ = VAE2d(in_chans=1, input_dims=(64, 64))
    _ = VAE2d(in_chans=2, input_dims=(64, 64))

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1.0, input_dims=(64, 64))
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1j, input_dims=(64, 64))


def test_in_chans_arg_values():
    """Test the values accepted by the ``in_chans`` arg."""
    # Should work with positive int
    _ = VAE2d(in_chans=1, input_dims=(64, 64))
    _ = VAE2d(in_chans=2, input_dims=(64, 64))

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=0, input_dims=(64, 64))

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=-1, input_dims=(64, 64))
