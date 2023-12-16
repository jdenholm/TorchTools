"""Test the arguments of the UNet model."""
import pytest


from torch_tools import UNet


def test_unet_in_chans_arg_type():
    """Test the types accepted by the ``in_chans`` argument."""
    # Should work with positive ints
    _ = UNet(in_chans=1, out_chans=1)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = UNet(in_chans=1.0, out_chans=1)

    with pytest.raises(TypeError):
        _ = UNet(in_chans=1j, out_chans=1)


def test_unet_in_chans_arg_values():
    """Test the values accepted by the ``in_chans`` argument."""
    # Should work with positive ints
    _ = UNet(in_chans=1, out_chans=1)
    _ = UNet(in_chans=2, out_chans=1)
    _ = UNet(in_chans=3, out_chans=1)

    # Should break with ints < 1
    with pytest.raises(ValueError):
        _ = UNet(in_chans=0, out_chans=1)

    with pytest.raises(ValueError):
        _ = UNet(in_chans=-1, out_chans=1)


def test_unet_out_chans_arg_type():
    """Test the types accepted by the ``out_chans`` argument."""
    # Should work with positive ints
    _ = UNet(in_chans=1, out_chans=1)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1.0)

    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1j)


def test_unet_out_chans_arg_values():
    """Test the values accepted by the ``out_chans`` argument."""
    # Should work with positive ints
    _ = UNet(in_chans=1, out_chans=1)
    _ = UNet(in_chans=1, out_chans=2)
    _ = UNet(in_chans=1, out_chans=3)

    # Should break with ints < 1
    with pytest.raises(ValueError):
        _ = UNet(in_chans=1, out_chans=0)

    with pytest.raises(ValueError):
        _ = UNet(in_chans=1, out_chans=-1)


def test_unet_features_start_arg_type():
    """Test the types accepted by the ``features_start`` argument."""
    # Should work with positive ints
    _ = UNet(in_chans=1, out_chans=1)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1, features_start=16.0)

    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1, features_start=32j)


def test_unet_features_start_arg_values():
    """Test the values accepted by the ``features_start`` argument."""
    # Should work with positive ints
    _ = UNet(in_chans=1, out_chans=1, features_start=1)
    _ = UNet(in_chans=1, out_chans=2, features_start=2)
    _ = UNet(in_chans=1, out_chans=3, features_start=3)

    # Should break with ints < 1
    with pytest.raises(ValueError):
        _ = UNet(in_chans=1, out_chans=1, features_start=0)

    with pytest.raises(ValueError):
        _ = UNet(in_chans=1, out_chans=1, features_start=-1)


def test_unet_num_layers_arg_types():
    """Test the types accepted by the ``num_layers`` argument type."""
    # Should work with ints of 2 or more
    _ = UNet(in_chans=1, out_chans=1, num_layers=2)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1, num_layers=2.0)

    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1, num_layers=2j)


def test_unet_num_layers_arg_values():
    """Test the values accepted by the ``num_layers`` argument."""
    # Should work with ints of 2 or more
    _ = UNet(in_chans=1, out_chans=1, num_layers=2)
    _ = UNet(in_chans=1, out_chans=1, num_layers=3)
    _ = UNet(in_chans=1, out_chans=1, num_layers=4)

    # Should break with ints < 2
    with pytest.raises(ValueError):
        _ = UNet(in_chans=1, out_chans=1, num_layers=1)

    with pytest.raises(ValueError):
        _ = UNet(in_chans=1, out_chans=1, num_layers=0)

    with pytest.raises(ValueError):
        _ = UNet(in_chans=1, out_chans=1, num_layers=-1)


def test_unet_pool_style_arg_types():
    """Test the types accepted by the ``pool_style`` arg."""
    _ = UNet(in_chans=1, out_chans=1, pool_style="max")

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1, pool_style=123)

    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1, pool_style=["max"])


def test_unet_pool_style_arg_values():
    """Test the values accepted by the ``pool_style`` arg."""
    _ = UNet(in_chans=1, out_chans=1, pool_style="max")
    _ = UNet(in_chans=1, out_chans=1, pool_style="avg")

    with pytest.raises(KeyError):
        _ = UNet(in_chans=1, out_chans=1, pool_style="Bullroarer Took")


def test_unet_bilinear_arg_types():
    """Test the types accepted by the ``bilinear`` arg."""
    _ = UNet(in_chans=1, out_chans=1, bilinear=True)
    _ = UNet(in_chans=1, out_chans=1, bilinear=False)

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1, bilinear=1)

    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1, bilinear=123j)


def test_unet_lr_slope_argument_types():
    """Test the types accepted by the ``lr_slope`` arg type."""
    _ = UNet(in_chans=1, out_chans=1, lr_slope=0.0)
    _ = UNet(in_chans=1, out_chans=1, lr_slope=0.1)

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1, lr_slope=0)

    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1, lr_slope=0j)


def test_unet_kernel_size_argument_types():
    """Test the types accepted by the ``kernel_size`` argument,"""
    # Should work with int
    _ = UNet(in_chans=1, out_chans=1, kernel_size=1)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1, kernel_size=1.0)

    with pytest.raises(TypeError):
        _ = UNet(in_chans=1, out_chans=1, kernel_size=1j)


def test_unet_kernel_size_argument_values():
    """Test the values accepted by the ``kernel_size`` argument."""
    # Should work with positive, odd, int
    _ = UNet(in_chans=1, out_chans=1, kernel_size=1)
    _ = UNet(in_chans=1, out_chans=1, kernel_size=3)
    _ = UNet(in_chans=1, out_chans=1, kernel_size=5)

    # Should break with ints < 1, or even positives
    for size in [-2, -1, 0, 2, 4]:
        with pytest.raises(ValueError):
            _ = UNet(in_chans=1, out_chans=1, kernel_size=size)


def test_unet_block_style_argument_values():
    """Test the values accepted by the ``block_style`` argument."""
    # Should work with allowed values
    for style in ["conv_res", "double_conv"]:
        _ = UNet(in_chans=1, out_chans=1, block_style=style)

    for bad_style in ["Denathor", 666]:
        with pytest.raises(ValueError):
            _ = UNet(in_chans=1, out_chans=1, block_style=bad_style)
