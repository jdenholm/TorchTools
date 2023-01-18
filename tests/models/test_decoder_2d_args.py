"""Test the arguments of torch_tools.models._decoder_2d.Decoder2d."""

import pytest

from torch_tools import Decoder2d


def test_in_chans_arg_types():
    """Test the types accepted by the `in_chans` arg."""
    # Should work with positive ints which divide by 2 (num_blocks - 1) times
    Decoder2d(
        in_chans=8,
        out_chans=3,
        num_blocks=4,
        bilinear=False,
        lr_slope=0.666,
    )

    # Should break with non-int
    with pytest.raises(TypeError):
        Decoder2d(
            in_chans=8.0,
            out_chans=3,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
        )
    # Should break with non-int
    with pytest.raises(TypeError):
        Decoder2d(
            in_chans=8j,
            out_chans=3,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
        )


def test_in_chans_arg_values():
    """Test the values accepted by the `in_chans` arg."""
    # Should work with positive ints which divide by 2 (num_blocks - 1) times
    Decoder2d(
        in_chans=8,
        out_chans=3,
        num_blocks=4,
        bilinear=False,
        lr_slope=0.666,
    )

    # Should break if 2 doesn't divide in_chans (num_blocks - 1) times.
    with pytest.raises(ValueError):
        Decoder2d(
            in_chans=1,
            out_chans=3,
            num_blocks=2,
            bilinear=False,
            lr_slope=0.666,
        )

    with pytest.raises(ValueError):
        Decoder2d(
            in_chans=30,
            out_chans=3,
            num_blocks=3,
            bilinear=False,
            lr_slope=0.666,
        )
