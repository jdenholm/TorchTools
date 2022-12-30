"""Test the call behaviour of blocks in `torch_tools.models._blocks_1d`."""

from torch import rand

from torch_tools.models._blocks_1d import DenseBlock, InputBlock


def test_dense_block_call_returns_correct_shape():
    """Test `DenseBlock` returns ouputs of the correct shape."""
    model = DenseBlock(
        in_feats=123,
        out_feats=321,
        batch_norm=True,
        dropout_prob=0.5,
        negative_slope=0.1,
    )

    assert model(rand(10, 123)).shape == (10, 321)


def test_dense_block_call_returns_correct_shape_no_bnorm():
    """Test `DenseBlock` returns ouputs of the correct shape."""
    model = DenseBlock(
        in_feats=123,
        out_feats=321,
        batch_norm=False,
        dropout_prob=0.5,
        negative_slope=0.1,
    )

    assert model(rand(10, 123)).shape == (10, 321)


def test_dense_block_call_returns_correct_shape_no_dropout():
    """Test `DenseBlock` returns ouputs of the correct shape."""
    model = DenseBlock(
        in_feats=123,
        out_feats=321,
        batch_norm=True,
        dropout_prob=0.0,
        negative_slope=0.1,
    )

    assert model(rand(10, 123)).shape == (10, 321)


def test_dense_block_call_returns_correct_shape_no_bnorm_no_dropout():
    """Test `DenseBlock` returns ouputs of the correct shape."""
    model = DenseBlock(
        in_feats=123,
        out_feats=321,
        batch_norm=False,
        dropout_prob=0.0,
        negative_slope=0.1,
    )

    assert model(rand(10, 123)).shape == (10, 321)


def test_input_block_call_returns_correct_shape():
    """Test `InputBlock` returns outputs of the correct shape."""
    model = InputBlock(in_feats=123, batch_norm=True, dropout=0.5)
    assert model(rand(10, 123)).shape == (10, 123)


def test_input_block_call_returns_correct_shape_no_dropout():
    """Test `InputBlock` returns outputs of the correct shape."""
    model = InputBlock(in_feats=123, batch_norm=True, dropout=0.0)
    assert model(rand(10, 123)).shape == (10, 123)


def test_input_block_call_returns_correct_shape_no_bnorm():
    """Test `InputBlock` returns outputs of the correct shape."""
    model = InputBlock(in_feats=123, batch_norm=False, dropout=0.5)
    assert model(rand(10, 123)).shape == (10, 123)


def test_input_block_call_returns_correct_shape_no_bnorm_no_dropout():
    """Test `InputBlock` returns outputs of the correct shape."""
    model = InputBlock(in_feats=123, batch_norm=False, dropout=0.0)
    assert model(rand(10, 123)).shape == (10, 123)
