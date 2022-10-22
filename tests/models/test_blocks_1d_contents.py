"""Test the contents of blocks in `torch_tools.models._blocks1d.`"""
# pylint: disable=protected-access

from torch.nn import Linear, BatchNorm1d, LeakyReLU, Dropout

from torch_tools.models._blocks_1d import DenseBlock, InputBlock


def test_contents_of_dense_block_when_full():
    """Test the contents of `DenseBlock` with everything in it."""
    block = DenseBlock(
        10,
        2,
        batch_norm=True,
        dropout_prob=0.5,
        final_block=False,
    )

    msg = "There should be four layers in the block."
    assert len(block._fwd_seq) == 4, msg

    msg = "First layer should be linear."
    assert isinstance((block._fwd_seq[0]), Linear), msg

    msg = "Second layer should be batch norm."
    assert isinstance(block._fwd_seq[1], BatchNorm1d)

    msg = "Third layer should be dropout."
    assert isinstance(block._fwd_seq[2], Dropout)

    msg = "Final layer should be leaky relu."
    assert isinstance(block._fwd_seq[3], LeakyReLU)


def test_contents_of_dense_block_with_no_batchnorm():
    """Test the contents of `DenseBlock` without a batchnorm layer."""
    block = DenseBlock(
        10,
        2,
        batch_norm=False,
        dropout_prob=0.5,
        final_block=False,
    )

    msg = "There should be three layers in the block."
    assert len(block._fwd_seq) == 3, msg

    msg = "First layer should be linear."
    assert isinstance((block._fwd_seq[0]), Linear), msg

    msg = "Second layer should be dropout."
    assert isinstance(block._fwd_seq[1], Dropout)

    msg = "Final layer should be leaky relu."
    assert isinstance(block._fwd_seq[2], LeakyReLU)


def test_contents_of_dense_block_with_no_dropout():
    """Test the contents of `DenseBlock` without Dropout."""
    block = DenseBlock(
        10,
        2,
        batch_norm=True,
        dropout_prob=0.0,
        final_block=False,
    )

    msg = "There should be three layers in the block."
    assert len(block._fwd_seq) == 3, msg

    msg = "First layer should be linear."
    assert isinstance((block._fwd_seq[0]), Linear), msg

    msg = "Second layer should be batch norm."
    assert isinstance(block._fwd_seq[1], BatchNorm1d)

    msg = "Final layer should be leaky relu."
    assert isinstance(block._fwd_seq[2], LeakyReLU)


def test_contents_of_dense_block_as_final_block():
    """Test the contents of `DenseBlock` with `final_block` True."""
    block = DenseBlock(10, 2, final_block=True)

    msg = "There should be one layersin the block."
    assert len(block._fwd_seq) == 1, msg

    msg = "First layer should be linear."
    assert isinstance((block._fwd_seq[0]), Linear), msg


def test_dropout_probability():
    """Test the dropout probability value is correctly assigned."""
    block = DenseBlock(10, 2, dropout_prob=0.1234)
    assert block._fwd_seq[2].p == 0.1234, "Dropout prob not correct."

    block = DenseBlock(10, 2, dropout_prob=0.987654321)
    assert block._fwd_seq[2].p == 0.987654321, "Dropout prob not correct."


def test_leaky_relu_slope_value_assignment():
    """Test the slope of the leaky relu layer."""
    block = DenseBlock(10, 2, negative_slope=0.1234)
    assert block._fwd_seq[3].negative_slope == 0.1234, "Slope not correct."

    block = DenseBlock(10, 2, negative_slope=0.98765432)
    assert block._fwd_seq[3].negative_slope == 0.98765432, "Slope not correct."


def test_contents_of_input_block_when_full():
    """Test the contents of `InputBlock` when full."""
    block = InputBlock(in_feats=10, batch_norm=True, dropout=0.5)

    # There should be three layers in the block
    assert len(block._fwd_seq) == 2, "There should be 2 layers in the block."

    # The first layer should be a batchnorm
    msg = "First layer should be batchnorm."
    assert isinstance(block._fwd_seq[0], BatchNorm1d), msg

    # The second layer should be a dropout
    msg = "Second layer should be a dropout."
    assert isinstance(block._fwd_seq[1], Dropout), msg


def test_contents_of_input_block_with_batchnorm_only():
    """Test the contents of `InputBlock` with just a batch-norm."""
    block = InputBlock(in_feats=10, batch_norm=True, dropout=0.0)

    # There should be three layers in the block
    assert len(block._fwd_seq) == 1, "There should be 1 layer in the block."

    # The first layer should be a batchnorm
    msg = "First layer should be batchnorm."
    assert isinstance(block._fwd_seq[0], BatchNorm1d), msg


def test_contents_of_input_block_with_dropout_only():
    """Test the contents of `InputBlock` with just a dropout."""
    block = InputBlock(in_feats=10, batch_norm=False, dropout=0.5)

    # There should be three layers in the block
    assert len(block._fwd_seq) == 1, "There should be 2 layers in the block."

    # The second layer should be a dropout
    msg = "Second layer should be a dropout."
    assert isinstance(block._fwd_seq[0], Dropout), msg


def test_input_block_dropout_prob():
    """Test assingment of the dropout probability `InputBlock`."""
    block = InputBlock(in_feats=10, batch_norm=True, dropout=0.1234)
    assert block._fwd_seq[1].p == 0.1234, "Unexpected dropout prob."

    block = InputBlock(in_feats=10, batch_norm=True, dropout=0.4321)
    assert block._fwd_seq[1].p == 0.4321, "Unexpected dropout prob."
