"""Tests for the arguments of `torch_tools.models._encoder_2d.Encoder2d`."""

import pytest

from torch_tools import Encoder2d


def test_in_chans_argument_type():
    """Test the types accepted by the `in_chans` argument."""
    # Should work with ints of one or more
    _ = Encoder2d(
        in_chans=1,
        start_features=64,
        num_blocks=4,
        pool_style="max",
        lr_slope=0.1,
    )

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=1.0,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
        )

    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=1j,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
        )


def test_in_chans_argument_values():
    """Test the values accepted bu the `in_chans` argument."""
    # Should work with ints of one or more
    _ = Encoder2d(
        in_chans=1,
        start_features=64,
        num_blocks=4,
        pool_style="max",
        lr_slope=0.1,
    )

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = Encoder2d(
            in_chans=0,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
        )

    with pytest.raises(ValueError):
        _ = Encoder2d(
            in_chans=-1,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
        )


def test_start_features_arg_types():
    """Test the types accepted by the `start_features` argument."""
    # Should work with ints of one or more
    _ = Encoder2d(
        in_chans=32,
        start_features=1,
        num_blocks=4,
        pool_style="max",
        lr_slope=0.1,
    )

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=1.0,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
        )
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=1j,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
        )


def test_start_features_argument_values():
    """Test the values accepted by the `start_features` argument."""
    # Should work with ints of one or more
    _ = Encoder2d(
        in_chans=32,
        start_features=1,
        num_blocks=4,
        pool_style="max",
        lr_slope=0.1,
    )

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = Encoder2d(
            in_chans=32,
            start_features=0,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
        )

    with pytest.raises(ValueError):
        _ = Encoder2d(
            in_chans=32,
            start_features=-1,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
        )


def test_num_blocks_argument_types():
    """Test the types accepted by the `num_blocks` argument."""
    # Should work with ints of two or more
    _ = Encoder2d(
        in_chans=32,
        start_features=64,
        num_blocks=2,
        pool_style="max",
        lr_slope=0.1,
    )

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=1.0,
            pool_style="max",
            lr_slope=0.1,
        )
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=1j,
            pool_style="max",
            lr_slope=0.1,
        )


def test_num_blocks_argument_values():
    """Test the values accepted by the `num_blocks` argument."""
    # Should work with ints of two or more
    _ = Encoder2d(
        in_chans=32,
        start_features=64,
        num_blocks=2,
        pool_style="max",
        lr_slope=0.1,
    )

    # Should break with ints less than 2
    with pytest.raises(ValueError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=1,
            pool_style="max",
            lr_slope=0.1,
        )
    with pytest.raises(ValueError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=0,
            pool_style="max",
            lr_slope=0.1,
        )


def test_pool_style_argument_type():
    """Test the types accepted by the `pool_style` argument."""
    # Should work with allowed strings
    _ = Encoder2d(
        in_chans=32,
        start_features=64,
        num_blocks=4,
        pool_style="max",
        lr_slope=0.1,
    )
    _ = Encoder2d(
        in_chans=32,
        start_features=64,
        num_blocks=4,
        pool_style="avg",
        lr_slope=0.1,
    )

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=4,
            pool_style=12345,
            lr_slope=0.1,
        )
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=4,
            pool_style=3.14,
            lr_slope=0.1,
        )


def test_pool_style_argument_values():
    """Test the values accepted by the `pool_style` argument."""
    # Should work with allowed strings
    _ = Encoder2d(
        in_chans=32,
        start_features=64,
        num_blocks=4,
        pool_style="max",
        lr_slope=0.1,
    )
    _ = Encoder2d(
        in_chans=32,
        start_features=64,
        num_blocks=4,
        pool_style="avg",
        lr_slope=0.1,
    )

    with pytest.raises(KeyError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=4,
            pool_style="Fatty Bolger",
            lr_slope=0.1,
        )


def test_lr_slope_arg_types():
    """Test the types accepted by the `lr_slope` argument."""
    # Should work with floats
    _ = Encoder2d(
        in_chans=32,
        start_features=64,
        num_blocks=4,
        pool_style="max",
        lr_slope=0.1,
    )

    # Should break with non-floats
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=1,
        )
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=1j,
        )
