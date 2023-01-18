"""Test the arguments of torch_tools.models._decoder_2d.Decoder2d."""

import pytest

from torch_tools import Decoder2d


def test_in_chans_arg_types():
    """Test the types accepted by the `in_chans` arg."""
    # Should work with positive ints which divide by 2 (num_blocks - 1) times
    _ = Decoder2d(
        in_chans=8,
        out_chans=3,
        num_blocks=4,
        bilinear=False,
        lr_slope=0.666,
    )

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8.0,
            out_chans=3,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
        )
    # Should break with non-int
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8j,
            out_chans=3,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
        )


def test_in_chans_arg_values():
    """Test the values accepted by the `in_chans` arg."""
    # Should work with positive ints which divide by 2 (num_blocks - 1) times
    _ = Decoder2d(
        in_chans=8,
        out_chans=3,
        num_blocks=4,
        bilinear=False,
        lr_slope=0.666,
    )

    # Should break if 2 doesn't divide in_chans (num_blocks - 1) times.
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=1,
            out_chans=3,
            num_blocks=2,
            bilinear=False,
            lr_slope=0.666,
        )

    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=30,
            out_chans=3,
            num_blocks=3,
            bilinear=False,
            lr_slope=0.666,
        )

    # Should break if less than 1
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=0,
            out_chans=3,
            num_blocks=3,
            bilinear=False,
            lr_slope=0.666,
        )
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=-128,
            out_chans=3,
            num_blocks=3,
            bilinear=False,
            lr_slope=0.666,
        )


def test_out_chans_arg_types():
    """Test the types accepted by the `out_chans` arg."""
    # Should work with ints of 1 or more
    _ = Decoder2d(
        in_chans=8,
        out_chans=1,
        num_blocks=4,
        bilinear=False,
        lr_slope=0.666,
    )

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=1.0,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
        )
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=1j,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
        )


def test_out_chans_arg_values():
    """Test the values accepted by the `out_chans` arg."""
    # Should work with ints of 1 or more
    _ = Decoder2d(
        in_chans=8,
        out_chans=1,
        num_blocks=4,
        bilinear=False,
        lr_slope=0.666,
    )

    # Should break with ints less than 1
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=0,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
        )
    # Should break with ints less than 1
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=-1,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
        )


def test_num_blocks_arg_types():
    """Test the types accepted by the `num_blocks` arg."""
    # Should work with ints of 1 or more
    _ = Decoder2d(
        in_chans=8,
        out_chans=3,
        num_blocks=1,
        bilinear=False,
        lr_slope=0.666,
    )

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=1.0,
            bilinear=False,
            lr_slope=0.666,
        )


def test_num_blocks_arg_values():
    """Test the values accepted by the `num_blocks` arg."""
    # Should work with ints of 1 or more
    _ = Decoder2d(
        in_chans=8,
        out_chans=3,
        num_blocks=1,
        bilinear=False,
        lr_slope=0.666,
    )

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=0,
            bilinear=False,
            lr_slope=0.666,
        )
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=-1,
            bilinear=False,
            lr_slope=0.666,
        )


def test_bilinear_arg_types():
    """Test the types accepted by the `bilinear` arg."""
    # Should work with bool
    _ = Decoder2d(
        in_chans=8,
        out_chans=3,
        num_blocks=1,
        bilinear=True,
        lr_slope=0.666,
    )
    _ = Decoder2d(
        in_chans=8,
        out_chans=3,
        num_blocks=1,
        bilinear=False,
        lr_slope=0.666,
    )

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=1,
            bilinear=1,
            lr_slope=0.666,
        )

    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=1,
            bilinear="Barliman Butterbur",
            lr_slope=0.666,
        )


def test_lr_slope_arg_types():
    """Test the types accepted by the `lr_slope` arg."""
    # Should work with floats
    _ = Decoder2d(
        in_chans=8,
        out_chans=3,
        num_blocks=3,
        bilinear=True,
        lr_slope=0.666,
    )

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=1,
            bilinear=True,
            lr_slope=1,
        )

    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=1,
            bilinear=True,
            lr_slope=1,
        )
