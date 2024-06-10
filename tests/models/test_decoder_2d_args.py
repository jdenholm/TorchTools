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
        kernel_size=3,
    )

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8.0,
            out_chans=3,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
        )
    # Should break with non-int
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8j,
            out_chans=3,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
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
        kernel_size=3,
    )

    # Should break if 2 doesn't divide in_chans (num_blocks - 1) times.
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=1,
            out_chans=3,
            num_blocks=2,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
        )

    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=30,
            out_chans=3,
            num_blocks=3,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
        )

    # Should break if less than 1
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=0,
            out_chans=3,
            num_blocks=3,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
        )
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=-128,
            out_chans=3,
            num_blocks=3,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
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
        kernel_size=3,
    )

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=1.0,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
        )
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=1j,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
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
        kernel_size=3,
    )

    # Should break with ints less than 1
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=0,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
        )
    # Should break with ints less than 1
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=-1,
            num_blocks=4,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
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
        kernel_size=3,
    )

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=1.0,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
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
        kernel_size=3,
    )

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=0,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
        )
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=-1,
            bilinear=False,
            lr_slope=0.666,
            kernel_size=3,
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
        kernel_size=3,
    )
    _ = Decoder2d(
        in_chans=8,
        out_chans=3,
        num_blocks=1,
        bilinear=False,
        lr_slope=0.666,
        kernel_size=3,
    )

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=1,
            bilinear=1,
            lr_slope=0.666,
            kernel_size=3,
        )

    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=1,
            bilinear="Barliman Butterbur",
            lr_slope=0.666,
            kernel_size=3,
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
        kernel_size=3,
    )

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=1,
            bilinear=True,
            lr_slope=1,
            kernel_size=3,
        )

    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=1,
            bilinear=True,
            lr_slope=1,
            kernel_size=3,
        )


def test_kernel_size_argument_types():
    """Test the types accepted by the ``kernel_size`` argument."""
    # Should work with ints
    _ = Decoder2d(
        in_chans=8,
        out_chans=3,
        num_blocks=3,
        bilinear=True,
        lr_slope=0.666,
        kernel_size=3,
    )

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=3,
            bilinear=True,
            lr_slope=0.666,
            kernel_size=3.0,
        )

    with pytest.raises(TypeError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=3,
            bilinear=True,
            lr_slope=0.666,
            kernel_size=3j,
        )


def test_kernel_size_argument_values():
    """Test the values accepted by the ``kernel_size`` arg."""
    # Should work with odd, positive ints
    for size in [1, 3, 5, 7]:
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=3,
            bilinear=True,
            lr_slope=0.666,
            kernel_size=size,
        )

    # Should break with positive evens and ints less than one
    for size in [-1, -2, 0, 2, 4, 6]:
        with pytest.raises(ValueError):
            _ = Decoder2d(
                in_chans=8,
                out_chans=3,
                num_blocks=3,
                bilinear=True,
                lr_slope=0.666,
                kernel_size=size,
            )


def test_min_up_feats_arg_types():
    """Test the types accepted by the ``min_up_feats`` arg."""
    # Should work with positive ints, or None
    for min_feats in [1, None]:
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=3,
            bilinear=True,
            lr_slope=0.666,
            kernel_size=3,
            min_up_feats=min_feats,
        )

    # Should break with non-int or non-None

    for min_feats in [1.0, 2.0j]:
        with pytest.raises(TypeError):
            _ = Decoder2d(
                in_chans=8,
                out_chans=3,
                num_blocks=3,
                bilinear=True,
                lr_slope=0.666,
                kernel_size=3,
                min_up_feats=min_feats,
            )


def test_min_up_feats_arg_values():
    """Test the values accepted by the ``min_up_feats`` argument."""
    # Should work with positive ints, or None
    for min_feats in [1, None]:
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=3,
            bilinear=True,
            lr_slope=0.666,
            kernel_size=3,
            min_up_feats=min_feats,
        )

    # Should break with non-positive ints
    for min_feats in [-2, -1, 0]:
        with pytest.raises(ValueError):
            _ = Decoder2d(
                in_chans=8,
                out_chans=3,
                num_blocks=3,
                bilinear=True,
                lr_slope=0.666,
                kernel_size=3,
                min_up_feats=min_feats,
            )


def test_decoder_2d_block_style_arg_values():
    """Test the types accepted by the ``block_style`` arg."""
    # Should work with allowed options
    for block_style in ["double_conv", "conv_res"]:
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=3,
            bilinear=True,
            lr_slope=0.666,
            kernel_size=3,
            block_style=block_style,
        )

    # Should break with any other option
    for block_style in [666, 1.0]:
        with pytest.raises(ValueError):
            _ = Decoder2d(
                in_chans=8,
                out_chans=3,
                num_blocks=3,
                bilinear=True,
                lr_slope=0.666,
                kernel_size=3,
                block_style=block_style,
            )


def test_in_chans_is_more_than_min_up_feats():
    """Make sure ``in_chans`` is more than ``min_up_feats``."""
    # Should work if ``in_chans`` >= ``min_up_feats``.
    _ = Decoder2d(
        in_chans=32,
        out_chans=3,
        num_blocks=3,
        bilinear=True,
        lr_slope=0.666,
        kernel_size=3,
        min_up_feats=8,
    )

    _ = Decoder2d(
        in_chans=16,
        out_chans=3,
        num_blocks=3,
        bilinear=True,
        lr_slope=0.666,
        kernel_size=3,
        min_up_feats=16,
    )

    # Should break if ``in_chans`` < ``min_up_feats``.
    with pytest.raises(ValueError):
        _ = Decoder2d(
            in_chans=8,
            out_chans=3,
            num_blocks=3,
            bilinear=True,
            lr_slope=0.666,
            kernel_size=3,
            min_up_feats=16,
        )
