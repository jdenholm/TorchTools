"""Test the call behaviour of `ConvNet2d`."""

import pytest

from torch import rand  # pylint: disable=no-name-in-module

from torch_tools.models import ConvNet2d


def test_with_correct_input_channels():
    """Test the model accepts the correct number of input channels."""
    # Test with one channel
    model = ConvNet2d(out_feats=2, encoder_style="resnet18", in_channels=1)
    _ = model(rand(10, 1, 50, 50))

    # Test with three channels
    model = ConvNet2d(out_feats=2, encoder_style="resnet18", in_channels=3)
    _ = model(rand(10, 3, 50, 50))

    # Test with five channels
    model = ConvNet2d(out_feats=2, encoder_style="resnet18", in_channels=5)
    _ = model(rand(10, 5, 50, 50))


def test_with_incorrect_input_channels():
    """Test the model breaks with the wrong number of input channels."""
    # Ask for one input channel, give three
    model = ConvNet2d(out_feats=2, encoder_style="resnet18", in_channels=1)
    with pytest.raises(RuntimeError):
        _ = model(rand(10, 3, 50, 50))

    # Ask for three inputs channels, give five
    model = ConvNet2d(out_feats=2, encoder_style="resnet18", in_channels=3)
    with pytest.raises(RuntimeError):
        _ = model(rand(10, 5, 50, 50))


def test_output_features_are_correct():
    """Test the model produces the correct number of output features."""
    # Ask for one output channel
    model = ConvNet2d(out_feats=1)
    preds = model(rand(10, 3, 50, 50))
    assert preds.shape == (10, 1), "Expected one output features."

    # Ask for 128 output features
    model = ConvNet2d(out_feats=128)
    preds = model(rand(10, 3, 50, 50))
    assert preds.shape == (10, 128), "Expected 128 output features."

    # Ask for 321 output features
    model = ConvNet2d(out_feats=321)
    preds = model(rand(10, 3, 50, 50))
    assert preds.shape == (10, 321), "Expected 321 output features."


def test_call_works_with_all_kinds_of_pools():
    """Test the forward function works with each pool option."""
    # With average pool
    model = ConvNet2d(out_feats=1, pool_style="avg")
    _ = model(rand(10, 3, 50, 50))

    model = ConvNet2d(out_feats=1, pool_style="max")
    _ = model(rand(10, 3, 50, 50))

    model = ConvNet2d(out_feats=1, pool_style="avg-max-concat")
    _ = model(rand(10, 3, 50, 50))
