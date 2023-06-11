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
        kernel_size=3,
    )

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=1.0,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=3,
        )

    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=1j,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=3,
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
        kernel_size=3,
    )

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = Encoder2d(
            in_chans=0,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=3,
        )

    with pytest.raises(ValueError):
        _ = Encoder2d(
            in_chans=-1,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=3,
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
        kernel_size=3,
    )

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=1.0,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=3,
        )
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=1j,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=3,
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
        kernel_size=3,
    )

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = Encoder2d(
            in_chans=32,
            start_features=0,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=3,
        )

    with pytest.raises(ValueError):
        _ = Encoder2d(
            in_chans=32,
            start_features=-1,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=3,
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
        kernel_size=3,
    )

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=1.0,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=3,
        )
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=1j,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=3,
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
        kernel_size=3,
    )

    # Should break with ints less than 2
    with pytest.raises(ValueError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=1,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=3,
        )
    with pytest.raises(ValueError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=0,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=3,
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
        kernel_size=3,
    )
    _ = Encoder2d(
        in_chans=32,
        start_features=64,
        num_blocks=4,
        pool_style="avg",
        lr_slope=0.1,
        kernel_size=3,
    )

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=4,
            pool_style=12345,
            lr_slope=0.1,
            kernel_size=3,
        )
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=4,
            pool_style=3.14,
            lr_slope=0.1,
            kernel_size=3,
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
        kernel_size=3,
    )
    _ = Encoder2d(
        in_chans=32,
        start_features=64,
        num_blocks=4,
        pool_style="avg",
        lr_slope=0.1,
        kernel_size=3,
    )

    with pytest.raises(KeyError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=4,
            pool_style="Fatty Bolger",
            lr_slope=0.1,
            kernel_size=3,
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
        kernel_size=3,
    )

    # Should break with non-floats
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=1,
            kernel_size=3,
        )
    with pytest.raises(TypeError):
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=1j,
            kernel_size=3,
        )


def test_kernel_size_argument_types():
    """Test the types accepted by the ``kernel_size`` argument."""
    # Should work with ints
    _ = Encoder2d(
        in_chans=32,
        start_features=64,
        num_blocks=4,
        pool_style="max",
        lr_slope=0.1,
        kernel_size=3,
    )

    # Should break with non-int
    for bad_size in [3.0, 3j, "Barliman Butterbur"]:
        with pytest.raises(TypeError):
            _ = Encoder2d(
                in_chans=32,
                start_features=64,
                num_blocks=4,
                pool_style="max",
                lr_slope=0.1,
                kernel_size=bad_size,
            )


def test_kernel_size_argument_values():
    """Test the values accepted by the ``kernel_size`` argument."""
    # Should work with positive, odd, ints
    for good_size in [1, 3, 5, 7]:
        _ = Encoder2d(
            in_chans=32,
            start_features=64,
            num_blocks=4,
            pool_style="max",
            lr_slope=0.1,
            kernel_size=good_size,
        )

    # Should break with even positive ints, and ints less than 1
    for bad_size in [-1, -2, 0, 2, 4, 6]:
        with pytest.raises(ValueError):
            _ = Encoder2d(
                in_chans=32,
                start_features=64,
                num_blocks=4,
                pool_style="max",
                lr_slope=0.1,
                kernel_size=bad_size,
            )
