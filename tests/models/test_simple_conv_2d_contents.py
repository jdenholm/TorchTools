"""Test the contents of ``torch_tools.SimpleConvNet2d``."""

from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d, Flatten, Linear
from torch.nn import AvgPool2d, MaxPool2d

from torch_tools import SimpleConvNet2d, Encoder2d
from torch_tools.models._adaptive_pools_2d import _ConcatMaxAvgPool2d


def test_simple_conv_net_contents():
    """Test the contents of ``SimpleConvNet2d``."""
    model = SimpleConvNet2d(in_chans=123, out_feats=8, adaptive_pool="avg")

    assert len(model) == 4
    assert isinstance(model[0], Encoder2d)
    assert isinstance(model[1], AdaptiveAvgPool2d)
    assert isinstance(model[2], Flatten)
    assert isinstance(model[3], Linear)


def test_adaptive_pools():
    """Test the adaptive pools are assigned correctly."""
    model = SimpleConvNet2d(in_chans=123, out_feats=8, adaptive_pool="avg")
    assert isinstance(model[1], AdaptiveAvgPool2d)

    model = SimpleConvNet2d(in_chans=123, out_feats=8, adaptive_pool="max")
    assert isinstance(model[1], AdaptiveMaxPool2d)

    model = SimpleConvNet2d(
        in_chans=123,
        out_feats=8,
        adaptive_pool="avg-max-concat",
    )
    assert isinstance(model[1], _ConcatMaxAvgPool2d)


def test_down_pool_assignment_with_avg():
    """Test the down-sampling pools are correctly set to average."""
    model = SimpleConvNet2d(in_chans=123, out_feats=8, downsample_pool="avg")
    encoder = model[0]
    for down_block in list(encoder.children())[1:]:
        assert isinstance(down_block[0], AvgPool2d)


def test_down_pool_assignment_with_max():
    """Test the down-sampling pools are correctly set to max."""
    model = SimpleConvNet2d(in_chans=123, out_feats=8, downsample_pool="max")
    encoder = model[0]
    for down_block in list(encoder.children())[1:]:
        assert isinstance(down_block[0], MaxPool2d)


def test_linear_layer_input_feats_with_max_pool():
    """Test the linear layer's input features with max pool.

    Notes
    -----
    The number of inputs features the linear layers takes should be
    ``features_start * 2 ** (num_blocks - 1)``.

    """
    model = SimpleConvNet2d(
        in_chans=123,
        out_feats=8,
        adaptive_pool="max",
        num_blocks=3,
        features_start=64,
    )

    linear_layer = model[3]

    assert linear_layer.in_features == (2**2) * 64


def test_linear_layer_input_feats_with_avg_pool():
    """Test the linear layer's input features with avg pool.

    Notes
    -----
    The number of inputs features the linear layers takes should be
    ``features_start * 2 ** (num_blocks - 1)``.

    """
    model = SimpleConvNet2d(
        in_chans=123,
        out_feats=8,
        adaptive_pool="avg",
        num_blocks=3,
        features_start=64,
    )

    linear_layer = model[3]

    assert linear_layer.in_features == (2**2) * 64


def test_linear_layer_input_feats_with_avg_max_concat_pool():
    """Test the linear layer's input features with avg-max-concat pool.

    Notes
    -----
    The number of inputs features the linear layers takes should be
    ``2 * features_start * 2 ** (num_blocks - 1)``. The extra factor of two
    is because we are using the "avg-max-concat" adaptive pool.

    """
    model = SimpleConvNet2d(
        in_chans=123,
        out_feats=8,
        adaptive_pool="avg-max-concat",
        num_blocks=3,
        features_start=64,
    )

    linear_layer = model[3]

    assert linear_layer.in_features == 2 * (2**2) * 64


def test_linear_layer_output_features():
    """Test the number of output features produced by the linear layer."""
    model = SimpleConvNet2d(in_chans=3, out_feats=8)
    assert model[3].out_features == 8

    model = SimpleConvNet2d(in_chans=3, out_feats=123)
    assert model[3].out_features == 123

    model = SimpleConvNet2d(in_chans=3, out_feats=321)
    assert model[3].out_features == 321
