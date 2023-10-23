"""Tests for the arguments of ``torch_tools.AutoEncoder2d``."""
import pytest

from torch_tools import AutoEncoder2d


def test_in_chans_arg_types():
    """Test the types accepted by the ``in_chans`` argument."""
    # Should work with positive ints
    _ = AutoEncoder2d(in_chans=1, out_chans=3)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=1.0, out_chans=3)

    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=1j, out_chans=3)


def test_in_chans_arg_values():
    """Test the values accepted by the ``in_chans`` argument."""
    # Should work with positive ints
    _ = AutoEncoder2d(in_chans=1, out_chans=3)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=0, out_chans=3)

    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=-1, out_chans=3)


def test_out_chans_arg_types():
    """Test the types accepted by the ``out_chans`` argument."""
    # Should work with positive ints
    _ = AutoEncoder2d(in_chans=3, out_chans=1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=1.0)

    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=1j)


def test_out_chans_argument_values():
    """Test the values accepted by the ``out_chans`` argument."""
    # Should work with positive ints
    _ = AutoEncoder2d(in_chans=3, out_chans=1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=3, out_chans=0)

    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=3, out_chans=-1)


def test_num_layers_argument_types():
    """Test the types accepted by the ``num_layers`` argument."""
    # Should work with ints of 2 or more
    _ = AutoEncoder2d(in_chans=3, out_chans=3, num_layers=2)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, num_layers=2.0)

    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, num_layers=2j)


def test_num_layers_arguments_values():
    """Test the values accepted by the ``num_layers`` argument."""
    # Should work with ints of 2 or more
    _ = AutoEncoder2d(in_chans=3, out_chans=3, num_layers=2)

    # Should break with ints less than two
    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, num_layers=1)

    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, num_layers=0)

    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, num_layers=-1)


def test_features_start_argument_types():
    """Test the types accepted by the ``features_start`` argument."""
    # Should work with ints of one or more
    _ = AutoEncoder2d(in_chans=3, out_chans=3, features_start=1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, features_start=1.0)

    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, features_start=1j)


def test_features_start_argument_values():
    """Test the values accepted by the ``features_start`` argument."""
    # Should work with ints of one or more
    _ = AutoEncoder2d(in_chans=3, out_chans=3, features_start=1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, features_start=0)

    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, features_start=-1)


def test_lr_slope_argument_types():
    """Test the types accepted by the ``lr_slope`` argument type."""
    # Should work with floats
    _ = AutoEncoder2d(in_chans=3, out_chans=3, lr_slope=0.1)

    # Should break with non-floats
    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, lr_slope=1)

    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, lr_slope=0.1j)


def test_pool_style_argument_types():
    """Test the types accepted by the ``pool_style`` argument type."""
    _ = AutoEncoder2d(in_chans=3, out_chans=3, pool_style="max")
    _ = AutoEncoder2d(in_chans=3, out_chans=3, pool_style="avg")

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, pool_style=1)

    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, pool_style=["Gandalf"])


def test_pool_style_argument_values():
    """Test the values accepted by the ``pool_style`` argument type."""
    _ = AutoEncoder2d(in_chans=3, out_chans=3, pool_style="max")
    _ = AutoEncoder2d(in_chans=3, out_chans=3, pool_style="avg")

    # Should break with not-allowed option
    with pytest.raises(KeyError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, pool_style="Thranduil")


def test_bilinear_argument_type():
    """Test the types accepted by the ``bilinear`` argument."""
    # Should work with bool
    _ = AutoEncoder2d(in_chans=3, out_chans=3, bilinear=True)
    _ = AutoEncoder2d(in_chans=3, out_chans=3, bilinear=False)

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, bilinear=1)

    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, bilinear=1j)


def test_kernel_size_argument_type():
    """Test the types accepted by the ``kernel_size`` argument."""
    # Should work with int
    _ = AutoEncoder2d(in_chans=3, out_chans=3, kernel_size=1)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, kernel_size=1.0)

    with pytest.raises(TypeError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, kernel_size=1j)


def test_kernel_size_argument_values():
    """Test the values accepted by the ``kernel_size`` argument."""
    # Should work with positive, odd, int
    _ = AutoEncoder2d(in_chans=3, out_chans=3, kernel_size=1)
    _ = AutoEncoder2d(in_chans=3, out_chans=3, kernel_size=3)
    _ = AutoEncoder2d(in_chans=3, out_chans=3, kernel_size=5)

    # Should break with ints less than one, or positive evens
    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, kernel_size=-1)

    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, kernel_size=0)

    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, kernel_size=2)

    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, kernel_size=4)


def test_block_style_argument_values():
    """Test the values accepted by the ``block_stlye`` argument."""
    # Should work with accepted values
    _ = AutoEncoder2d(in_chans=3, out_chans=3, block_style="double_conv")
    _ = AutoEncoder2d(in_chans=3, out_chans=3, block_style="conv_res")

    # Should break with any other value
    with pytest.raises(ValueError):
        _ = AutoEncoder2d(in_chans=3, out_chans=3, block_style="Gandalf")
