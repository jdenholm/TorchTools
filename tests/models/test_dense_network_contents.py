"""Test the contents of `torch-tools.models.DenseNetwork`."""

from torch.nn import Dropout, BatchNorm1d

from torch_tools.models import DenseNetwork


# pylint: disable=protected-access


def test_full_input_block_contents():
    """Test contents of input block with `BatchNorm1d` and `Dropout`."""
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        input_bnorm=True,
        input_dropout=0.25,
    )

    in_block = model._input_block

    # There should be two layers in the input block
    assert len(in_block._fwd_seq) == 2, "Expected two layers in input block."

    # The first layer in the input block should be BatchNorm1d
    msg = "First layer should be BatchNorm1d."
    assert isinstance(in_block._fwd_seq[0], BatchNorm1d), msg

    # The second layer in the input block should be Dropout.
    msg = "Second layer should be Dropout"
    assert isinstance(in_block._fwd_seq[1], Dropout), msg


def test_input_block_contents_with_batchnorm_only():
    """Test contents of input block when user asks for batchnorm only.

    Notes
    -----
    By setting the argument `input_dropout=0.0`, we exclude the dropout
    layer.

    """
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        input_bnorm=True,
        input_dropout=0.0,
    )

    in_block = model._input_block

    # There should only be one layer in the input block
    assert len(in_block._fwd_seq) == 1, "Expected one layer in input block."

    # The only layer in input block should be a BatchNorm1d
    msg = "First layer should be BatchNorm1d"
    assert isinstance(in_block._fwd_seq[0], BatchNorm1d), msg


def test_input_block_contents_with_dropout_only():
    """Test the contents of input block when user asks for dropout only.

    Notes
    -----
    To ask for input dropout only, we set `input_bnorm=False`.

    """
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        input_bnorm=False,
        input_dropout=0.25,
    )

    in_block = model._input_block

    # There should only be one layer in the input block
    assert len(in_block._fwd_seq) == 1, "Expected one layer in input block."

    # The only layer in input block should be a Dropout
    msg = "First layer should be BatchNorm1d"
    assert isinstance(in_block._fwd_seq[0], Dropout), msg


def test_input_block_dropout_probability_assignment():
    """Test the dropout probability in input block is correctly assigned."""
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        input_bnorm=True,
        input_dropout=0.123456
    )
    input_dropout_prob = model._input_block._fwd_seq[1].p
    assert input_dropout_prob == 0.123456, "Dropout prob incorrectly assigned."

    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        input_bnorm=True,
        input_dropout=0.654321
    )
    input_dropout_prob = model._input_block._fwd_seq[1].p
    assert input_dropout_prob == 0.654321, "Dropout prob incorrectly assigned."
