"""Test the call behaviour of ``torch_tools.AutoEncoder2d``."""
from itertools import product
import pytest

from torch import rand  # pylint: disable=no-name-in-module


from torch_tools import AutoEncoder2d


def test_autoencoder_2d_return_channels():
    """Test the correct number of channels are returned.

    Sweeps through all of the upsampling and pooling combinations and tests.

    """
    for bilinear, pool in product([True, False], ["max", "avg"]):
        model = AutoEncoder2d(
            in_chans=3,
            out_chans=123,
            bilinear=bilinear,
            pool_style=pool,
        )
        assert model(rand(10, 3, 64, 64)).shape[1] == 123

        model = AutoEncoder2d(
            in_chans=123,
            out_chans=222,
            bilinear=bilinear,
            pool_style=pool,
        )
        assert model(rand(10, 123, 64, 64)).shape[1] == 222


def test_autoencoder_2d_return_sizes():
    """Test the size of the output batch is correct.

    Sweeps through all of the pool--upsampling combinations and tests.

    """
    for bilinear, pool in product([True, False], ["max", "avg"]):
        model = AutoEncoder2d(
            in_chans=3,
            out_chans=3,
            bilinear=bilinear,
            pool_style=pool,
        )
        assert model(rand(10, 3, 64, 128)).shape == (10, 3, 64, 128)

        model = AutoEncoder2d(
            in_chans=9,
            out_chans=3,
            bilinear=bilinear,
            pool_style=pool,
        )
        assert model(rand(10, 9, 128, 64)).shape == (10, 3, 128, 64)

        model = AutoEncoder2d(
            in_chans=15,
            out_chans=3,
            bilinear=bilinear,
            pool_style=pool,
        )
        assert model(rand(10, 15, 32, 32)).shape == (10, 3, 32, 32)
