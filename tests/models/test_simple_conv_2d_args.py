"""Test the types and values of the arguments of ``SimpleConvNet2d``."""
import pytest

from torch_tools import SimpleConvNet2d


def test_in_chans_argument_type():
    """Test the types accepted by ``in_chans`` argument type."""
    # Should work with ints of 1 or more
    _ = SimpleConvNet2d(in_chans=1, out_feats=8)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=1.0, out_feats=8)
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=1j, out_feats=8)


def test_in_chans_argument_value_type():
    """Test the values accepted by the ``in_chans`` arguments type."""
    # Should work with ints of 1 or more
    _ = SimpleConvNet2d(in_chans=1, out_feats=8)

    # Should break with ints less than 1
    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=0, out_feats=8)
    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=-1, out_feats=8)


def test_out_chans_argument_types():
    """Test the types accepted by the ``out_chans`` argument type."""
    # Should work with ints of 1 or more
    _ = SimpleConvNet2d(in_chans=64, out_feats=1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=16, out_feats=1.0)
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=16, out_feats=1j)


def test_out_chans_argument_values():
    """Test the values accepted by the ``out_chans`` argument."""
    # Should work with ints of 1 or more
    _ = SimpleConvNet2d(in_chans=64, out_feats=1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=16, out_feats=0)
    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=16, out_feats=-1)


def test_features_start_argument_type():
    """Test the types accepted by the ``features_start`` argument."""
    # Should work with ints of 1 or more
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, features_start=1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, features_start=1.0)

    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, features_start=1j)


def test_features_start_argument_values():
    """Test the values accepted by the ``features_start`` argument."""
    # Should work with ints of 1 or more
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, features_start=1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, features_start=0)
    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, features_start=-1)


def test_num_blocks_argument_types():
    """Test the types accepted by the ``num_blocks`` argument."""
    # Should work with ints of 2 or more
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, num_blocks=2)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, num_blocks=1.0)
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, num_blocks=1j)


def test_num_blocks_argument_values():
    """Test the values accepted by the ``num_blocks`` argument."""
    # Should work with ints of 2 or more
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, num_blocks=2)

    # Should break with ints less than 2
    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, num_blocks=1)
    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, num_blocks=0)
    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, num_blocks=-1)


def test_downsample_pool_argument_type():
    """Test the types accepted by the ``downsample_pool`` argument type."""
    # Should work with accepted str
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, downsample_pool="max")

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, downsample_pool=123)
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, downsample_pool=[123])


def test_downsample_pool_argument_values():
    """Test the values accepted by the ``downsample_pool`` argument value."""
    # Should work either ``"max"`` or ``"avg"``
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, downsample_pool="max")
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, downsample_pool="avg")

    # Should break with any other string
    with pytest.raises(KeyError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, downsample_pool="Gimli")


def test_adaptive_pool_argument_types():
    """Test the types accepted by the ``adaptive_pool`` argument type."""
    # Should work either ``"max"``, ``"avg"`` or ``"avg-max-concat"``
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, adaptive_pool="max")
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, adaptive_pool="avg")
    _ = SimpleConvNet2d(
        in_chans=64,
        out_feats=8,
        adaptive_pool="avg-max-concat",
    )

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, adaptive_pool=123)
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, adaptive_pool=1j)


def test_adaptive_pool_argument_values():
    """Test the values accepted by the ``adaptive_pool`` argument type."""
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, adaptive_pool="max")
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, adaptive_pool="avg")
    _ = SimpleConvNet2d(
        in_chans=64,
        out_feats=8,
        adaptive_pool="avg-max-concat",
    )

    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, adaptive_pool="Thorin")
    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, adaptive_pool="Thrain")


def test_lr_slope_argument_types():
    """Test the types accepted the ``lr_slope`` argument."""
    # Should work with floats
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, lr_slope=0.1)

    # Should break with non-floats
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, lr_slope=1)
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, lr_slope=1.0j)


def test_kernel_size_argument_types():
    """Test the types accepted by the ``kernel_size`` argument."""
    # Should work with int
    _ = SimpleConvNet2d(in_chans=123, out_feats=321, kernel_size=1)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=123, out_feats=321, kernel_size=1.0)

    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=123, out_feats=321, kernel_size=1j)


def test_kernel_size_argument_values():
    """Test the values accepted by the ``kernel_size`` argument."""
    # Should work with positive, odd, ints
    _ = SimpleConvNet2d(in_chans=123, out_feats=321, kernel_size=1)
    _ = SimpleConvNet2d(in_chans=123, out_feats=321, kernel_size=3)
    _ = SimpleConvNet2d(in_chans=123, out_feats=321, kernel_size=5)

    # Should break with any other ints
    for size in [-2, -1, 0, 2, 4]:
        with pytest.raises(ValueError):
            _ = SimpleConvNet2d(in_chans=123, out_feats=321, kernel_size=size)


def test_block_style_argument_values():
    """Test the values accepted by the ``block_size`` argument."""
    _ = SimpleConvNet2d(in_chans=1, out_feats=64, block_style="double_conv")
    _ = SimpleConvNet2d(in_chans=1, out_feats=64, block_style="conv_res")

    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=1, out_feats=64, block_style=666)

    with pytest.raises(ValueError):
        _ = SimpleConvNet2d(in_chans=1, out_feats=64, block_style="Arwen")
