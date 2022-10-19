"""Test for blocks in `torch_tools.models.._blocks_1d`."""
import pytest

from torch_tools.models._blocks_1d import DenseBlock


def test_dense_block_in_feats_types():
    """Test types accepted for `in_feats` argument of `DenseBlock`."""
    # Should work with ints of one or more.
    _ = DenseBlock(in_feats=10, out_feats=2)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats=10.0, out_feats=2)
    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats="10", out_feats=2)
    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats=10j, out_feats=2)


def test_dense_block_in_feats_values():
    """Test values for `in_feats` argument of `DenseBlock`."""
    # Should work with ints of one or more.
    _ = DenseBlock(in_feats=1, out_feats=2)
    _ = DenseBlock(in_feats=2, out_feats=2)

    # Should break with ints less than 1
    with pytest.raises(ValueError):
        _ = DenseBlock(in_feats=0, out_feats=2)
    with pytest.raises(ValueError):
        _ = DenseBlock(in_feats=-1, out_feats=2)


def test_dense_block_out_feats_types():
    """Test the types of the `out_feats` argument of `DenseBlock`."""
    # Should work with ints of one or more
    _ = DenseBlock(in_feats=10, out_feats=1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats=10, out_feats=2.0)
    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats=10, out_feats="1")
    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats=10, out_feats=1j)


def test_dense_block_out_feats_values():
    """Test the values accepted by the `out_feats` argument of `DenseBlock`."""
    # Should work with ints of one or more
    _ = DenseBlock(in_feats=10, out_feats=1)
    _ = DenseBlock(in_feats=10, out_feats=2)

    # Should break with ints less than 1
    with pytest.raises(ValueError):
        _ = DenseBlock(in_feats=10, out_feats=0)
    with pytest.raises(ValueError):
        _ = DenseBlock(in_feats=10, out_feats=-1)


def test_dense_blocks_batch_norm_arg_types():
    """Test types accepted by `batch_norm` arg of `DenseBlock`."""
    # Should work with bool
    _ = DenseBlock(in_feats=10, out_feats=1, batch_norm=True)
    _ = DenseBlock(in_feats=10, out_feats=2, batch_norm=False)

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats=10, out_feats=1, batch_norm=1)
    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats=10, out_feats=1, batch_norm="True")
    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats=10, out_feats=1, batch_norm=10.0)
