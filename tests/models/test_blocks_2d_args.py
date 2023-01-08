"""Test the arguments accepted by blocks in `torch_tools.models._blocks_2d`."""

import pytest

from torch_tools.models._blocks_2d import ConvBlock, DoubleConvBlock, ResBlock
from torch_tools.models._blocks_2d import UNetUpBlock, DownBlock, UpBlock


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


def test_conv_block_out_chans_arg_types():
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


def test_conv_block_leaky_relu_argument_types():
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


def test_conv_block_lr_slope_argument_types():
    """Test the `lr_slope` argument type."""
    # Should work with float
    _ = ConvBlock(in_chans=2, out_chans=1, lr_slope=0.1)

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=2, out_chans=1, lr_slope=1)
    with pytest.raises(TypeError):
        _ = ConvBlock(in_chans=2, out_chans=1, lr_slope=1j)


def test_double_conv_block_in_chans_types():
    """Test the types accepted by the `in_chans` argument."""
    # Should work with ints of one or more
    _ = DoubleConvBlock(in_chans=1, out_chans=5, lr_slope=0.1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = DoubleConvBlock(in_chans=1.0, out_chans=5, lr_slope=0.1)
    with pytest.raises(TypeError):
        _ = DoubleConvBlock(in_chans=1.0j, out_chans=5, lr_slope=0.1)


def test_double_conv_block_in_chans_values():
    """Test the values accepted by the `in_chans` arg."""
    # Should work with ints of one or more
    _ = DoubleConvBlock(in_chans=1, out_chans=5, lr_slope=0.1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = DoubleConvBlock(in_chans=0, out_chans=10, lr_slope=0.1)
    with pytest.raises(ValueError):
        _ = DoubleConvBlock(in_chans=-1, out_chans=10, lr_slope=0.1)


def test_double_conv_block_out_chans_types():
    """Test the types accepted by the `out_chans` argument."""
    # Should work with ints of one or more.
    _ = DoubleConvBlock(in_chans=10, out_chans=1, lr_slope=0.1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = DoubleConvBlock(in_chans=10, out_chans=1.0, lr_slope=0.1)
    with pytest.raises(TypeError):
        _ = DoubleConvBlock(in_chans=10, out_chans=1.0j, lr_slope=0.1)


def test_double_conv_block_out_chans_arg_values():
    """Test the values accepted by the `out_chans` argument."""
    # Should work with ints of one or more
    _ = DoubleConvBlock(in_chans=10, out_chans=1, lr_slope=0.1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = DoubleConvBlock(in_chans=10, out_chans=0, lr_slope=0.1)
    with pytest.raises(ValueError):
        _ = DoubleConvBlock(in_chans=10, out_chans=-1, lr_slope=0.1)


def test_double_conv_lr_slope_argument_types():
    """Test the types accepted by the `lr_slope` arg."""
    # Should work with floats
    _ = DoubleConvBlock(in_chans=10, out_chans=2, lr_slope=0.0)
    _ = DoubleConvBlock(in_chans=10, out_chans=2, lr_slope=0.1)

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = DoubleConvBlock(in_chans=10, out_chans=2, lr_slope=1)
    with pytest.raises(TypeError):
        _ = DoubleConvBlock(in_chans=10, out_chans=2, lr_slope=1j)


def test_res_block_in_chans_arg_types():
    """Test the types accepted by the `in_chans` arg."""
    # Should work with ints of one or more
    _ = ResBlock(in_chans=1)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = ResBlock(in_chans=1.0)
    with pytest.raises(TypeError):
        _ = ResBlock(in_chans=1.0j)


def test_res_block_in_chans_arg_values():
    """Test the values accepted by the `in_chans` argument."""
    # Should work with ints of 1 or more
    _ = ResBlock(in_chans=1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = ResBlock(in_chans=0)
    with pytest.raises(ValueError):
        _ = ResBlock(in_chans=-1)


def test_unet_up_block_in_chans_arg_types():
    """Test the types accepted by the `in_chans` argument."""
    # Should work with ints of two or more
    _ = UNetUpBlock(in_chans=2, out_chans=1, bilinear=False, lr_slope=0.1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = UNetUpBlock(
            in_chans=2.0,
            out_chans=1,
            bilinear=False,
            lr_slope=0.1,
        )
    with pytest.raises(TypeError):
        _ = UNetUpBlock(
            in_chans=2j,
            out_chans=1,
            bilinear=False,
            lr_slope=0.1,
        )


def test_unet_up_block_in_chans_arg_values():
    """Test the values accepted by the `in_chans` argument."""
    # Should work with even ints of two or more
    _ = UNetUpBlock(in_chans=2, out_chans=1, bilinear=False, lr_slope=0.1)
    _ = UNetUpBlock(in_chans=4, out_chans=1, bilinear=False, lr_slope=0.1)

    # Should break with ints less than two
    with pytest.raises(ValueError):
        _ = UNetUpBlock(in_chans=1, out_chans=1, bilinear=False, lr_slope=0.1)
    with pytest.raises(ValueError):
        _ = UNetUpBlock(in_chans=0, out_chans=1, bilinear=False, lr_slope=0.1)

    # Should break with uneven ints
    with pytest.raises(ValueError):
        _ = UNetUpBlock(in_chans=3, out_chans=1, bilinear=False, lr_slope=0.1)
    with pytest.raises(ValueError):
        _ = UNetUpBlock(in_chans=3, out_chans=1, bilinear=False, lr_slope=0.1)


def test_unet_up_block_out_chans_arg_types():
    """Test the types accepted by the `out_chans` argument."""
    # Should work with ints of one or more
    _ = UNetUpBlock(in_chans=2, out_chans=1, bilinear=False, lr_slope=0.1)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = UNetUpBlock(
            in_chans=2,
            out_chans=1.0,
            bilinear=False,
            lr_slope=0.1,
        )
    with pytest.raises(TypeError):
        _ = UNetUpBlock(
            in_chans=2,
            out_chans=1j,
            bilinear=False,
            lr_slope=0.1,
        )


def test_unet_up_block_out_chans_arg_values():
    """Test the values accepted by the `out_chans` arg."""
    # Should work with ints of one or more
    _ = UNetUpBlock(in_chans=2, out_chans=1, bilinear=False, lr_slope=0.1)

    # Should break with anything less than one
    with pytest.raises(ValueError):
        _ = UNetUpBlock(in_chans=2, out_chans=0, bilinear=False, lr_slope=0.1)
    with pytest.raises(ValueError):
        _ = UNetUpBlock(in_chans=2, out_chans=-1, bilinear=False, lr_slope=0.1)


def test_unet_upblock_bilinear_arg_types():
    """Test the types allowed by the `bilinear` arg."""
    # Should work with bool
    _ = UNetUpBlock(in_chans=2, out_chans=1, bilinear=True, lr_slope=0.1)
    _ = UNetUpBlock(in_chans=2, out_chans=1, bilinear=False, lr_slope=0.1)

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = UNetUpBlock(in_chans=2, out_chans=1, bilinear=1, lr_slope=0.1)
    with pytest.raises(TypeError):
        _ = UNetUpBlock(in_chans=2, out_chans=1, bilinear="True", lr_slope=0.1)


def test_unet_upblock_leaky_relu_arg_types():
    """Test the types accepted by the `lr_slope` argument."""
    # Should work with float
    _ = UNetUpBlock(in_chans=2, out_chans=1, bilinear=False, lr_slope=0.1)

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = UNetUpBlock(in_chans=2, out_chans=1, bilinear=False, lr_slope=1)
    with pytest.raises(TypeError):
        _ = UNetUpBlock(in_chans=2, out_chans=1, bilinear=False, lr_slope=1j)


def test_down_block_in_chans_arg_types():
    """Test the types accepted by the `in_chans` arg."""
    # Should work with ints of one or more
    _ = DownBlock(in_chans=1, out_chans=8, pool="max", lr_slope=0.1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = DownBlock(in_chans=1.0, out_chans=8, pool="max", lr_slope=0.1)
    with pytest.raises(TypeError):
        _ = DownBlock(in_chans=1j, out_chans=8, pool="max", lr_slope=0.1)


def test_down_block_in_chans_arg_values():
    """Test the values accepted by the `in_chans` arg."""
    # Should work with ints of one or more
    _ = DownBlock(in_chans=1, out_chans=8, pool="max", lr_slope=0.1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = DownBlock(in_chans=0, out_chans=8, pool="max", lr_slope=0.1)
    with pytest.raises(ValueError):
        _ = DownBlock(in_chans=-1, out_chans=8, pool="max", lr_slope=0.1)


def test_down_block_out_chans_arg_types():
    """Test the types accepted by the `out_chans` arg."""
    # Should work with ints of one or more
    _ = DownBlock(in_chans=8, out_chans=1, pool="max", lr_slope=0.1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = DownBlock(in_chans=8, out_chans=1.0, pool="max", lr_slope=0.1)
    with pytest.raises(TypeError):
        _ = DownBlock(in_chans=8, out_chans=1j, pool="max", lr_slope=0.1)


def test_down_block_out_chans_arg_values():
    """Test the values accepted by the `out_chans` arg."""
    # Should work with ints of one or more
    _ = DownBlock(in_chans=8, out_chans=1, pool="max", lr_slope=0.1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = DownBlock(in_chans=8, out_chans=0, pool="max", lr_slope=0.1)
    with pytest.raises(ValueError):
        _ = DownBlock(in_chans=8, out_chans=-1, pool="max", lr_slope=0.1)


def test_down_block_pool_arg_types():
    """Test the types accepted by the `pool` argument."""
    # Should work with allowed strings
    _ = DownBlock(in_chans=8, out_chans=1, pool="max", lr_slope=0.1)
    _ = DownBlock(in_chans=8, out_chans=1, pool="avg", lr_slope=0.1)

    # Should work non-str
    with pytest.raises(TypeError):
        _ = DownBlock(in_chans=8, out_chans=1, pool=1, lr_slope=0.1)
    with pytest.raises(TypeError):
        _ = DownBlock(in_chans=8, out_chans=1, pool=[], lr_slope=0.1)


def test_down_block_pool_arg_values():
    """Test the values accepted by the `pool` arg."""
    # Should work with allowed strings
    _ = DownBlock(in_chans=8, out_chans=1, pool="max", lr_slope=0.1)
    _ = DownBlock(in_chans=8, out_chans=1, pool="avg", lr_slope=0.1)

    # Should break with strings which are not allowed
    with pytest.raises(KeyError):
        _ = DownBlock(in_chans=8, out_chans=1, pool="Gandalf", lr_slope=0.1)
    with pytest.raises(KeyError):
        _ = DownBlock(in_chans=8, out_chans=1, pool="Saruman", lr_slope=0.1)


def test_down_block_lr_slope_arg_type():
    """Test the types accepted by the `lr_slope` argument."""
    # Should work with floats
    _ = DownBlock(in_chans=8, out_chans=1, pool="max", lr_slope=0.1)

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = DownBlock(in_chans=8, out_chans=1, pool="max", lr_slope=1)
    with pytest.raises(TypeError):
        _ = DownBlock(in_chans=8, out_chans=1, pool="max", lr_slope=1j)


def test_up_block_in_chans_arg_type():
    """Test the types accepted by the `in_chans` arg."""
    # Should work with ints of one or more
    _ = UpBlock(in_chans=1, out_chans=3, bilinear=True, lr_slope=0.1)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = UpBlock(in_chans=1.0, out_chans=1, bilinear=True, lr_slope=0.1)
    with pytest.raises(TypeError):
        _ = UpBlock(in_chans=1j, out_chans=1, bilinear=True, lr_slope=0.1)


def test_up_block_in_chans_arg_values():
    """Test the values accepts by the `in_chans` arg."""
    # Should work with ints of one or more
    _ = UpBlock(in_chans=1, out_chans=2, bilinear=True, lr_slope=0.1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = UpBlock(in_chans=0, out_chans=2, bilinear=True, lr_slope=0.1)
    with pytest.raises(ValueError):
        _ = UpBlock(in_chans=-1, out_chans=2, bilinear=True, lr_slope=0.1)


def test_up_block_out_chans_arg_type():
    """Test the types accepted by the `out_chans` argument."""
    # Should work with ints of one or more
    _ = UpBlock(in_chans=1, out_chans=1, bilinear=True, lr_slope=0.1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = UpBlock(in_chans=1, out_chans=1.0, bilinear=True, lr_slope=0.1)
    with pytest.raises(TypeError):
        _ = UpBlock(in_chans=1, out_chans=1j, bilinear=True, lr_slope=0.1)


def test_up_block_out_chans_arg_values():
    """Test the values accepted by the `out_chans` arg."""
    # Should work with ints of one or more
    _ = UpBlock(in_chans=1, out_chans=1, bilinear=True, lr_slope=0.1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = UpBlock(in_chans=1, out_chans=0, bilinear=True, lr_slope=0.1)
    with pytest.raises(ValueError):
        _ = UpBlock(in_chans=1, out_chans=-1, bilinear=True, lr_slope=0.1)


def test_up_block_bilinear_arg_types():
    """Test the types accepted by the `bilinear` arg."""
    # Should work with bool
    _ = UpBlock(in_chans=1, out_chans=1, bilinear=True, lr_slope=0.1)
    _ = UpBlock(in_chans=1, out_chans=1, bilinear=False, lr_slope=0.1)

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = UpBlock(in_chans=1, out_chans=1, bilinear=1, lr_slope=0.1)
    with pytest.raises(TypeError):
        _ = UpBlock(
            in_chans=1,
            out_chans=1,
            bilinear="Hamfast Gamgee",
            lr_slope=0.1,
        )


def test_up_block_lr_slope_arg_types():
    """Test the types accepted by the `lr_slope` argument."""
    # Should work with floats
    _ = UpBlock(in_chans=1, out_chans=1, bilinear=True, lr_slope=0.1)

    # Should break with non-floats
    with pytest.raises(TypeError):
        _ = UpBlock(in_chans=1, out_chans=1, bilinear=True, lr_slope=1)
    with pytest.raises(TypeError):
        _ = UpBlock(in_chans=1, out_chans=1, bilinear=True, lr_slope=1.0j)
