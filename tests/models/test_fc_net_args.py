"""Test the types and values of the arguments in `FCNet`."""
import pytest


from torch_tools import FCNet


def test_fc_net_in_feats_arg_types():
    """Test the tpyes accepted by the `in_feats` argument."""
    # Should work with ints of 1 or more.
    _ = FCNet(in_feats=1, out_feats=10)

    # Should break with non-int.
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=1.0, out_feats=10)
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=1j, out_feats=10)


def test_fc_net_in_feats_arg_values():
    """Test the values accepted by the `in_feats` argument."""
    # Should work with ints of 1 or more.
    _ = FCNet(in_feats=1, out_feats=10)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = FCNet(in_feats=0, out_feats=10)
    with pytest.raises(ValueError):
        _ = FCNet(in_feats=-1, out_feats=10)


def test_fc_net_out_feats_arg_types():
    """Test the types accepted by the `out_feats` argument."""
    # Should work with ints of 1 or more
    _ = FCNet(in_feats=10, out_feats=1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=1.0)
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=1j)


def test_fc_net_out_feats_arg_values():
    """Test the values accepted by the `out_feats` argument."""
    # Should work with ints of 1 or more
    _ = FCNet(in_feats=10, out_feats=1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = FCNet(in_feats=10, out_feats=0)
    with pytest.raises(ValueError):
        _ = FCNet(in_feats=10, out_feats=-1)


def test_fc_net_hidden_sizes_argument_types():
    """Test the types accepted by the `hidden_sizes` argument type."""
    # Should work with Tuple[int, ...] or None
    _ = FCNet(in_feats=10, out_feats=1, hidden_sizes=(5, 5))
    _ = FCNet(in_feats=10, out_feats=1, hidden_sizes=None)

    # Should break with non-tuple
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=1, hidden_sizes=[5, 5])

    # Should break if the tuple contains non-ints
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=1, hidden_sizes=(10, 1, 1.0))
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=1, hidden_sizes=(10, 10, 1j))


def test_fc_net_input_bnorm_arg_types():
    """Test the types accepted by the `input_bnorm` arg type."""
    # Should work with bool
    _ = FCNet(in_feats=10, out_feats=1, input_bnorm=True)
    _ = FCNet(in_feats=10, out_feats=1, input_bnorm=False)

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=1, input_bnorm=1)
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=1, input_bnorm="Gil-gallad")


def test_fc_net_input_dropout_arg_types():
    """Test the types accepted by the `input_dropout` arg."""
    # Should work with floats on [0, 1)
    _ = FCNet(in_feats=10, out_feats=2, input_dropout=0.0)
    _ = FCNet(in_feats=10, out_feats=2, input_dropout=0.999)

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=2, input_dropout=1)
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=2, input_dropout=0.5j)


def test_fc_net_input_dropout_arg_values():
    """Test the values accepted by the `input_dropout` arg."""
    # Should work with floats on [0, 1)
    _ = FCNet(in_feats=10, out_feats=2, input_dropout=0.0)
    _ = FCNet(in_feats=10, out_feats=2, input_dropout=0.999)

    # Should break with floats not on [0, 1)
    with pytest.raises(ValueError):
        _ = FCNet(in_feats=10, out_feats=2, input_dropout=-0.0001)
    with pytest.raises(ValueError):
        _ = FCNet(in_feats=10, out_feats=2, input_dropout=1.0)


def test_fc_net_hidden_bnorm_arg_types():
    """Test the types accepted by the `input_bnorm` arg."""
    # Should work with bool
    _ = FCNet(in_feats=10, out_feats=2, hidden_bnorm=True)
    _ = FCNet(in_feats=10, out_feats=2, hidden_bnorm=False)

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=2, hidden_bnorm=1)
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=2, hidden_bnorm=[True])


def test_fc_net_hidden_dropout_arg_types():
    """Test the types accepted by the `hidden_dropout` arg."""
    # Should work with float on [0, 1)
    _ = FCNet(in_feats=10, out_feats=2, hidden_dropout=0.0)
    _ = FCNet(in_feats=10, out_feats=2, hidden_dropout=0.999)

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=2, hidden_dropout=0)
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=2, hidden_dropout=0j)


def test_fc_net_hidden_dropout_arg_values():
    """Test values accepted by the `hidden_dropout` arg."""
    # Should work with float on [0, 1)
    _ = FCNet(in_feats=10, out_feats=2, hidden_dropout=0.0)
    _ = FCNet(in_feats=10, out_feats=2, hidden_dropout=0.999)

    # Should break with floats not on [0, 1)
    with pytest.raises(ValueError):
        _ = FCNet(in_feats=10, out_feats=2, hidden_dropout=-0.001)
    with pytest.raises(ValueError):
        _ = FCNet(in_feats=10, out_feats=2, hidden_dropout=1.0)


def test_fc_net_negative_slope_arg_types():
    """Test the types accepted by the `negative_slope` argument."""
    # Should work with floats
    _ = FCNet(in_feats=10, out_feats=2, negative_slope=0.0)
    _ = FCNet(in_feats=10, out_feats=2, negative_slope=0.1)

    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=2, negative_slope=0)
    with pytest.raises(TypeError):
        _ = FCNet(in_feats=10, out_feats=2, negative_slope=0j)
