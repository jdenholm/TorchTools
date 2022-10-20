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


def test_dense_block_batch_norm_arg_types():
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


def test_dense_block_dropout_prob_arg_type():
    """Type-check `dropout_prob` argument of `DenseBlock`."""
    # Should work with float on [0..0, 1.0)
    _ = DenseBlock(in_feats=10, out_feats=2, dropout_prob=0.0)
    _ = DenseBlock(in_feats=10, out_feats=2, dropout_prob=0.999)

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats=10, out_feats=2, dropout_prob=0)
    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats=10, out_feats=2, dropout_prob="1")


def test_dense_block_dropout_prob_values():
    """Value-check `dropouut_prob` argument of `DenseBlock`."""
    # Should work with float on [0.0, 1.0)
    _ = DenseBlock(in_feats=10, out_feats=2, dropout_prob=0.0)
    _ = DenseBlock(in_feats=10, out_feats=2, dropout_prob=0.999)

    # Should break with floats not on [0.0, 1.0)
    with pytest.raises(ValueError):
        _ = DenseBlock(in_feats=10, out_feats=2, dropout_prob=-0.00001)
    with pytest.raises(ValueError):
        _ = DenseBlock(in_feats=10, out_feats=2, dropout_prob=1.0)
    with pytest.raises(ValueError):
        _ = DenseBlock(in_feats=10, out_feats=2, dropout_prob=1.00001)


def test_dense_block_final_block_argument_type():
    """Type-check the `final_block` argument of `DenseBlock`."""
    # Should work with bool
    _ = DenseBlock(in_feats=10, out_feats=2, final_block=True)
    _ = DenseBlock(in_feats=10, out_feats=2, final_block=False)

    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats=10, out_feats=2, final_block=1)
    with pytest.raises(TypeError):
        _ = DenseBlock(in_feats=10, out_feats=2, final_block="True")
