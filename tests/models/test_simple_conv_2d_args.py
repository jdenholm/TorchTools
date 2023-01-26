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
    """test the types accepted by the ``downsample_pool`` argument type."""
    # Should work with accepted str
    _ = SimpleConvNet2d(in_chans=64, out_feats=8, downsample_pool="max")

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(in_chans=64, out_feats=8, downsample_pool=123)
