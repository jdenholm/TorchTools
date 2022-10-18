"""Tests for functions in `torch_tools.models._argument_processing`."""
import pytest


from torch_tools.models import _argument_processing as ap


def test_process_num_feats_types():
    """Test `ap.process_num_feats`'s type checking."""
    # Should work with int
    _ = ap.process_num_feats(1)

    # Should break with any non-int
    with pytest.raises(TypeError):
        _ = ap.process_num_feats(1.0)
    with pytest.raises(TypeError):
        _ = ap.process_num_feats("1.0")
    with pytest.raises(TypeError):
        _ = ap.process_num_feats(1.0j)


def test_process_num_feats_values():
    """test `ap.process_num_feats`'s value checking."""
    # Should work with ints of one and above
    _ = ap.process_num_feats(1)
    _ = ap.process_num_feats(2)

    # Should fail with anything less than one
    with pytest.raises(ValueError):
        _ = ap.process_num_feats(0)
    with pytest.raises(ValueError):
        _ = ap.process_num_feats(-1)


def test_process_boolean_arg_types():
    """Test `ap.process_boolean_arg`'s type checking."""
    # Should work with bool
    _ = ap.process_boolean_arg(True)
    _ = ap.process_boolean_arg(False)

    # Should fail with non-bool
    with pytest.raises(TypeError):
        _ = ap.process_boolean_arg(1)
    with pytest.raises(TypeError):
        _ = ap.process_boolean_arg(1.0)
    with pytest.raises(TypeError):
        _ = ap.process_boolean_arg("Batman")


def test_process_dropout_argument_types():
    """Test `ap.process_dropout_argument`'s type checking."""
    # Should work with floats on (0.0, 1.0]
    _ = ap.process_dropout_prob(0.0)
    _ = ap.process_dropout_prob(0.99)

    # Should fail any non-floats
    with pytest.raises(TypeError):
        _ = ap.process_dropout_prob(0)
    with pytest.raises(TypeError):
        _ = ap.process_dropout_prob("Hello")
    with pytest.raises(TypeError):
        _ = ap.process_dropout_prob(1.0j)


def test_process_dropout_argument_values():
    """Test `ap.process_dropout_argument`'s value checking."""
    # Should work with floats on [0.0, 1.0)
    _ = ap.process_dropout_prob(0.0)
    _ = ap.process_dropout_prob(0.5)
    _ = ap.process_dropout_prob(0.99)

    # Should fail with floats outwith [0.0,, 1.0)
    with pytest.raises(ValueError):
        _ = ap.process_dropout_prob(-0.00001)
    with pytest.raises(ValueError):
        _ = ap.process_dropout_prob(1.0)
    with pytest.raises(ValueError):
        _ = ap.process_dropout_prob(1.0001)
