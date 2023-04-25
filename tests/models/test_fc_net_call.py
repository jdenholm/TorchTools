"""Test calling `FCNet` with different input and output features."""

import pytest

from torch import rand  # pylint: disable=no-name-in-module

from torch_tools import FCNet


def test_model_with_correct_input_features():
    """Test the model works with different input features."""
    # With input batchnorm and dropout
    model = FCNet(10, 2, hidden_sizes=(5, 5, 5))
    _ = model(rand(2, 10))

    # Without input batchnorm and dropout
    model = FCNet(
        20,
        2,
        hidden_sizes=(5, 5, 5),
        input_bnorm=False,
        input_dropout=0.0,
    )
    _ = model(rand(2, 20))


def test_model_with_incorrect_input_features():
    """Test the model breaks when using the wrong number of input features."""
    # With input batchnorm and dropout
    model = FCNet(10, 2, hidden_sizes=(5, 5, 5))
    with pytest.raises(RuntimeError):
        _ = model(rand(2, 20))

    # Without input batchnorm and dropout
    model = FCNet(
        10,
        2,
        hidden_sizes=(5, 5, 5),
        input_bnorm=False,
        input_dropout=0.0,
    )
    with pytest.raises(RuntimeError):
        _ = model(rand(2, 20))


def test_number_of_output_features_is_correct():
    """Test the model produces the correct number of output features."""
    model = FCNet(10, 2, hidden_sizes=(5, 5, 5))
    assert model(rand(2, 10)).shape == (2, 2)

    model = FCNet(10, 128, hidden_sizes=(5, 5, 5))
    assert model(rand(2, 10)).shape == (2, 128)
