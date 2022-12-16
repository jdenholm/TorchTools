"""Test the arguments accepted by blocks in `torch_tools.models._blocks_2d`."""

import pytest

from torch_tools.models._blocks_2d import ConvBlock


def test_conv_block_in_chans_arg_types():
    """Test the `in_chans` argument types."""
    # Should work with positive ints greater than one
    _ = ConvBlock(in_chans=1, out_chans=2)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=1.0, out_chans=2)
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=1j, out_chans=2)


def test_conv_block_in_chans_arg_values():
    """Test the values accepted by the `in_chans` argument."""
    # Should work with positive ints greater than one.
    _ = ConvBlock(in_chans=1, out_chans=2)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = ConvBlock(in_chans=0, out_chans=2)
    with pytest.raises(ValueError):
        _ = ConvBlock(in_chans=-1, out_chans=2)


def test_conv_block_out_chans_arg_type():
    """Test the types accepted by the `out_chans` argument."""
    # Should work with positive ints greater than one
    _ = ConvBlock(in_chans=1, out_chans=1)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=1, out_chans=1.0)
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=1, out_chans=1j)


def test_conv_block_out_chans_arg_values():
    """Test the values accepted by the `out_chans` argument."""
    # Should work with positive ints greater than one
    _ = ConvBlock(in_chans=1, out_chans=1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = ConvBlock(in_chans=2, out_chans=0)
    with pytest.raises(ValueError):
        _ = ConvBlock(in_chans=2, out_chans=-1)


def test_conv_block_batch_norm_arg_types():
    """Test the `batch_norm` argument type."""
    # Should work with bool
    _ = ConvBlock(in_chans=2, out_chans=2, batch_norm=True)
    _ = ConvBlock(in_chans=2, out_chans=2, batch_norm=False)

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=2, out_chans=2, batch_norm=1)
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=2, out_chans=2, batch_norm="Shadowfax")
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=2, out_chans=2, batch_norm=0.0)


def test_conv_block_leaky_relu_argument_type():
    """Test the `leaky_relu` argument type."""
    # Should work with bool
    _ = ConvBlock(in_chans=2, out_chans=1, leaky_relu=True)
    _ = ConvBlock(in_chans=2, out_chans=1, leaky_relu=False)

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=2, out_chans=1, leaky_relu=1)
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=2, out_chans=1, leaky_relu="Bilbo Baggins")
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=2, out_chans=1, leaky_relu=1.0)


def test_conv_block_lr_slope_argument_type():
    """Test the `lr_slope` argument type."""
    # Should work with float
    _ = ConvBlock(in_chans=2, out_chans=1, lr_slope=0.1)

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=2, out_chans=1, lr_slope=1)
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=2, out_chans=1, lr_slope=1j)
