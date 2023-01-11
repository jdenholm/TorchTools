"""Test the contents of `torch_tools.models.DenseNetwork`."""

from torch.nn import Dropout, BatchNorm1d, Linear, LeakyReLU

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

    in_block = model[0]

    assert len(in_block) == 2, "Expected two layers in input block."

    assert isinstance(in_block[0], BatchNorm1d), "1st layer should be batchnorm."

    assert isinstance(in_block[1], Dropout), "2nd layer should be Dropout"


def test_input_block_contents_with_batchnorm_only():
    """Test contents of input block when user asks for batchnorm only.

    By setting the argument `input_dropout=0.0`, we exclude the dropout
    layer.

    """
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        input_bnorm=True,
        input_dropout=0.0,
    )

    in_block = model[0]

    assert len(in_block) == 1, "Expected one layer in input block."

    assert isinstance(in_block[0], BatchNorm1d), "1st layer should be batchnorm"


def test_input_block_contents_with_dropout_only():
    """Test the contents of input block when user asks for dropout only.

    To ask for input dropout only, we set `input_bnorm=False`.

    """
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        input_bnorm=False,
        input_dropout=0.25,
    )

    in_block = model[0]

    assert len(in_block) == 1, "Expected one layer in input block."

    assert isinstance(in_block[0], Dropout), "1st layer should be BatchNorm1d"


def test_input_batchnorm_number_of_feats_assignment():
    """Test the input batchnorm layer is assigned correct number of feats."""
    msg = "Unexpected number of batchnorm features."
    model = DenseNetwork(123, 2, input_bnorm=True)
    input_bnorm = model[0][0]
    assert input_bnorm.num_features == 123, msg

    msg = "Unexpected number of batchnorm features."
    model = DenseNetwork(321, 2, input_bnorm=True)
    input_bnorm = model[0][0]
    assert input_bnorm.num_features == 321, msg


def test_input_block_dropout_probability_assignment():
    """Test the dropout probability in input block is correctly assigned."""
    model = DenseNetwork(
        in_feats=10, out_feats=2, input_bnorm=True, input_dropout=0.123456
    )
    input_dropout_prob = model[0][1].p
    assert input_dropout_prob == 0.123456, "Dropout prob incorrectly assigned."

    model = DenseNetwork(
        in_feats=10, out_feats=2, input_bnorm=True, input_dropout=0.654321
    )
    input_dropout_prob = model[0][1].p
    assert input_dropout_prob == 0.654321, "Dropout prob incorrectly assigned."


def test_number_of_dense_blocks_with_no_hidden_layers():
    """Test the number of dense blocks in `DenseNetwork`.

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False,
        — input_dropout=0.0,

    """
    # With no hidden layers (hidden_sizes=None), the model should only have
    # one dense block.
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        hidden_sizes=None,
        input_bnorm=False,
        input_dropout=0.0,
    )

    dense_blocks = model
    msg = "Should only be one dense block when no hidden layers are requested."
    assert len(dense_blocks) == 1, msg


def test_number_of_hidden_blocks_with_one_hidden_layer():
    """Test the number of dense blocks with one hidden layer.

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0

    """
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        hidden_sizes=((5,)),
        input_bnorm=False,
        input_dropout=0.0,
    )

    # There should be two dense blocks if we ask for one hidden layer.
    # `print(model)` if that seems confusing.

    msg = "Should be two dense blocks when one hidden layer is requested."
    assert len(model) == 2, msg


def test_number_of_dense_blocks_with_seven_hidden_layers():
    """Test the numer of dense blocks with seven hidden layers.

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0

    """
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        hidden_sizes=(7 * (10,)),
        input_bnorm=False,
        input_dropout=0.0,
    )

    # There should be 8 dense block when 7 hidden layers are requested
    msg = "Should be 8 dense blocks when you ask for 7 hidden layers."
    assert len(model) == 8, msg


def test_linear_layer_sizes_in_dense_blocks_with_no_hidden_layers():
    """Test feature dimensions in the linear layers with no hidden layers.

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0

    """
    # Test with no hidden layers
    model = DenseNetwork(
        10,
        2,
        hidden_sizes=None,
        input_bnorm=False,
        input_dropout=0.0,
    )
    msg = "The linear layer should have 10 input features."
    assert model[0][0].in_features == 10, msg

    msg = "The linear layer should have 2 output features."
    assert model[0][0].out_features == 2, msg


def test_linear_layer_sizes_in_dense_blocks_with_hidden_layers():
    """Test the feature dimensions in the linear layers with hidden layers.

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0

    """
    # Test with hidden layers
    in_feats, out_feats = 128, 2
    hidden_sizes = (64, 32, 16, 8, 16, 32, 64)
    model = DenseNetwork(
        in_feats,
        out_feats,
        hidden_sizes=hidden_sizes,
        input_bnorm=False,
        input_dropout=0.0,
    )

    in_sizes = iter((in_feats,) + hidden_sizes)
    out_sizes = iter(hidden_sizes + (out_feats,))
    for _, module in model.named_children():

        msg = "Unexpected number of input features in linear layer."
        assert module[0].in_features == next(in_sizes), msg

        msg = "Unexpected number of output features in linear layer."
        assert module[0].out_features == next(out_sizes), msg


def test_hidden_block_contents_with_dropout_and_batchnorm():
    """Test contents of dense blocks with hidden layers, bnroms and dropout.

    The dense blocks should contain:

        Linear -> BatchNorm1d -> Dropout -> LeakyReLU

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0

    """
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        hidden_sizes=(5, 5, 5),
        hidden_dropout=0.5,
        hidden_bnorm=True,
        input_bnorm=False,
        input_dropout=0.0,
    )

    # The final block should only contain a Linear layer, so we chop it off
    non_final_blocks = list(model.named_children())[:-1]

    for _, block in non_final_blocks:

        msg = "Each non-final block should contain four layers."
        assert len(block) == 4, msg

        msg = "First layer of dense block should be Linear."
        assert isinstance(block[0], Linear), msg

        msg = "Second layer of dense block should be BatchNorm1d."
        assert isinstance(block[1], BatchNorm1d), msg

        msg = "Third layer of dense block should be Dropout."
        assert isinstance(block[2], Dropout), msg

        msg = "Fourth layer of dense block should be LeakyReLU."
        assert isinstance(block[3], LeakyReLU), msg


def test_hidden_block_contents_with_batchnorm_and_no_dropout():
    """Test contents of dense blocks with hidden layers and only batchnorm.

    The dense blocks should contain:

        Linear -> BatchNorm1d -> LeakyReLU

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0

    """
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        hidden_dropout=0.0,
        hidden_bnorm=True,
        input_bnorm=False,
        input_dropout=0.0,
    )

    non_final_blocks = list(model.named_children())[:-1]

    for _, block in non_final_blocks:

        msg = "Each non-final block should contain three layers."
        assert len(block) == 3, msg

        msg = "The first layer of the dense block should be a Linear."
        assert isinstance(block[0], Linear), msg

        msg = "The second layer of the dense block should be BatchNorm1d."
        assert isinstance(block[1], BatchNorm1d), msg

        msg = "The final layer of the dense block should be LeakyReLU."
        assert isinstance(block[2], LeakyReLU), msg


def test_hidden_block_contents_with_dropout_and_no_batchnorm():
    """Test contents of dense blocks with hidden layers and only dropout.

    The dense blocks should contain:

        Linear -> Dropout -> LeakyReLU

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0

    """
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        hidden_dropout=0.5,
        hidden_bnorm=False,
        input_bnorm=False,
        input_dropout=0.0,
    )

    non_final_blocks = list(model.named_children())[:-1]

    for _, block in non_final_blocks:

        msg = "There should be three layers in the block."
        assert len(block) == 3, msg

        msg = "The first layer of the dense block should be Linear."
        assert isinstance(block[0], Linear), msg

        msg = "The second layer of the dense block should be Dropout."
        assert isinstance(block[1], Dropout), msg

        msg = "The final layer of the dense block should be LeakyReLU."
        assert isinstance(block[2], LeakyReLU), msg


def test_hidden_block_contents_with_no_dropout_and_no_batch_norm():
    """Test contents of dense block's hidden layers no dropout/batchnorm.

    The contents of the dense blocks should be:

        Linear -> LeakyReLU

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0

    """
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        hidden_dropout=0.0,
        hidden_bnorm=False,
        input_bnorm=False,
        input_dropout=0.0,
    )

    non_final_blocks = list(model.named_children())[:-1]

    for _, block in non_final_blocks:

        msg = "The dense block should contain two layers."
        assert len(block) == 2, msg

        msg = "The first layer of the dense block should be a Linear."
        assert isinstance(block[0], Linear), msg

        msg = "The final layer of the dense block should be LeakyReLU."
        assert isinstance(block[1], LeakyReLU), msg


def test_hidden_batchnorm_number_of_features_are_correct():
    """Test number of features in the hidden layer batchnorms.

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0

    """
    hidden_sizes = (2, 4, 6, 8)
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        hidden_sizes=(2, 4, 6, 8),
        hidden_bnorm=True,
        hidden_dropout=0.25,
        input_bnorm=False,
        input_dropout=0.0,
    )

    non_final_blocks = list(model.named_children())[:-1]

    for (_, block), bnorm_feats in zip(non_final_blocks, hidden_sizes):

        msg = "Unexpected number of features in hidden block's batch norm."
        batch_norm = block[1]
        assert batch_norm.num_features == bnorm_feats, msg


def test_hidden_block_dropout_probability_assignment():
    """Test the dropout probability in thie hidden blocks is correct.

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0

    """
    model = DenseNetwork(
        in_feats=10,
        out_feats=2,
        hidden_sizes=(8, 6, 4),
        hidden_dropout=0.123456789,
        hidden_bnorm=True,
        input_bnorm=False,
        input_dropout=0.0,
    )

    non_final_blocks = list(model.named_children())[:-1]

    for _, block in non_final_blocks:

        dropout_layer = block[2]
        msg = "Unexpected dropout prob in hidden blocks."
        assert dropout_layer.p == 0.123456789, msg


def test_final_layer_feature_sizes_with_no_hidden_layers():
    """Test the feature sizes of the last linear layer in dense blocks.

    With no hidden layers, the feature sizes of the final (and only) linear
    layer should be equal to `in_feats` and `out_feats`.

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0


    """
    model = DenseNetwork(
        in_feats=987,
        out_feats=123,
        hidden_sizes=None,
        input_bnorm=False,
        input_dropout=0.0,
    )

    # There should only be one dense block
    assert len(model) == 1, "Should only be one layer in dense block."

    # There should only be one thing in the only dense block
    msg = "Dense block should have len 1."
    assert len(model[0]) == 1, msg

    # The linear layer in the only dense block should have `in_feats`
    msg = "Input features of only linear layer should be 987."
    assert model[0][0].in_features == 987, msg

    # The linear layer in the only dense block should have `out_feats`
    msg = "Output features of only linear layer should be 123."
    assert model[0][0].out_features == 123, msg


def test_final_linear_layer_feature_sizes_with_hidden_layers():
    """Test the feature sizes of the last linear layer in dense blocks.

    With hidden layers included, the final linear layer should have
    `hidden_sizes[-1]` input features and `out_feats` output features.

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0


    """
    in_feats, out_feats = 128, 2
    hidden_sizes = (63, 32, 666)
    model = DenseNetwork(
        in_feats=in_feats,
        out_feats=out_feats,
        hidden_sizes=hidden_sizes,
        input_bnorm=False,
        input_dropout=0.0,
    )

    final_linear_layer = model[-1][0]

    # The final linear layer should have hiddens_sizes[-1] input features
    msg = f"Expected {hidden_sizes[-1]} input features."
    assert final_linear_layer.in_features == hidden_sizes[-1], msg

    # The final linear layer should have `out_feats` out_features.
    msg = f"Expected {out_feats} output features."
    assert final_linear_layer.out_features == out_feats, msg


def test_leaky_relu_negative_slope_assignment():
    """Test the `negative_slope` value assignment of the leaky relu.

    Notes
    -----
    We also make sure there is no input block by setting:
        — input_bnorm=False
        — input_dropout=0.0


    """
    model = DenseNetwork(
        in_feats=100,
        out_feats=10,
        hidden_sizes=(50, 50),
        negative_slope=0.12345678,
        input_bnorm=False,
        input_dropout=0.0,
    )

    non_final_blocks = list(model.named_children())[:-1]
    for _, block in non_final_blocks:

        leaky_relu = block[-1]

        msg = "Wrong slope value in leaky relu."
        assert leaky_relu.negative_slope == 0.12345678, msg
