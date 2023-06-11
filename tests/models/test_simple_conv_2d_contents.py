"""Test the contents of ``torch_tools.SimpleConvNet2d``."""

import pytest

from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d, Flatten, Linear
from torch.nn import AvgPool2d, MaxPool2d, Conv2d, BatchNorm2d, LeakyReLU
from torch.nn import Module

from torch_tools import SimpleConvNet2d, Encoder2d
from torch_tools.models._adaptive_pools_2d import _ConcatMaxAvgPool2d
from torch_tools.models._fc_net import FCNet
from torch_tools.models._blocks_2d import ConvBlock, DoubleConvBlock
from torch_tools.models._blocks_2d import DownBlock


def kernel_size_check(block: Module, kernel_size: int):
    """Make sure any kernel sizes are correct."""
    if isinstance(block, Conv2d):
        assert block.kernel_size == (kernel_size, kernel_size)


def test_simple_conv_net_contents():
    """Test the contents of ``SimpleConvNet2d``."""
    model = SimpleConvNet2d(in_chans=123, out_feats=8, adaptive_pool="avg")

    assert len(model) == 4
    assert isinstance(model[0], Encoder2d)
    assert isinstance(model[1], AdaptiveAvgPool2d)
    assert isinstance(model[2], Flatten)
    assert isinstance(model[3], FCNet)


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

    fc_net = model[3]

    assert fc_net[0][0].in_features == (2**2) * 64


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

    fc_net = model[3]

    assert fc_net[0][0].in_features == (2**2) * 64


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

    fc_net = model[3]

    assert fc_net[0][0].in_features == 2 * (2**2) * 64


def test_linear_layer_output_features():
    """Test the number of output features produced by the linear layer."""
    model = SimpleConvNet2d(in_chans=3, out_feats=8)
    assert model[3][-1][-1].out_features == 8

    model = SimpleConvNet2d(in_chans=3, out_feats=123)
    assert model[3][-1][-1].out_features == 123

    model = SimpleConvNet2d(in_chans=3, out_feats=321)
    assert model[3][-1][-1].out_features == 321


def test_encoder_double_conv_contents():
    """Test the layer sizes at each block in the encoder."""
    model = SimpleConvNet2d(
        in_chans=3,
        out_feats=8,
        num_blocks=5,
        features_start=123,
        lr_slope=0.123456,
    )

    encoder = model[0]
    double_conv = encoder[0]
    assert isinstance(double_conv, DoubleConvBlock)

    for idx in range(2):
        assert isinstance(double_conv[idx], ConvBlock)
        assert isinstance(double_conv[idx][0], Conv2d)
        assert isinstance(double_conv[idx][1], BatchNorm2d)
        assert isinstance(double_conv[idx][2], LeakyReLU)

    assert double_conv[0][0].in_channels == 3
    assert double_conv[0][0].out_channels == 123
    assert double_conv[0][1].num_features == 123
    assert double_conv[0][2].negative_slope == 0.123456

    assert double_conv[1][0].in_channels == 123
    assert double_conv[1][0].out_channels == 123
    assert double_conv[1][1].num_features == 123
    assert double_conv[1][2].negative_slope == 0.123456


def test_encoder_down_block_contents_with_avg_pool():
    """Test the contents of the down blocks in the encoder."""
    model = SimpleConvNet2d(
        in_chans=3,
        out_feats=8,
        num_blocks=5,
        features_start=64,
        lr_slope=0.654321,
        downsample_pool="avg",
    )

    encoder = model[0]

    in_chans = 64

    for down_block in list(encoder.children())[1:]:
        assert isinstance(down_block, DownBlock)
        assert isinstance(down_block[0], AvgPool2d)
        assert isinstance(down_block[1], DoubleConvBlock)

        # Test the contents of the first conv block in the down block
        assert isinstance(down_block[1][0], ConvBlock)
        assert isinstance(down_block[1][0][0], Conv2d)
        assert isinstance(down_block[1][0][1], BatchNorm2d)
        assert isinstance(down_block[1][0][2], LeakyReLU)

        assert down_block[1][0][0].in_channels == in_chans
        assert down_block[1][0][0].out_channels == in_chans * 2
        assert down_block[1][0][1].num_features == in_chans * 2
        assert down_block[1][0][2].negative_slope == 0.654321

        # Test the contents of the first conv block in the down block
        assert isinstance(down_block[1][1], ConvBlock)
        assert isinstance(down_block[1][1][0], Conv2d)
        assert isinstance(down_block[1][1][1], BatchNorm2d)
        assert isinstance(down_block[1][1][2], LeakyReLU)

        assert down_block[1][1][0].in_channels == in_chans * 2
        assert down_block[1][1][0].out_channels == in_chans * 2
        assert down_block[1][1][1].num_features == in_chans * 2
        assert down_block[1][1][2].negative_slope == 0.654321

        in_chans *= 2


def test_simple_conv_net_2d_contents_with_variable_kernel_sizes():
    """Test the contents with different kernel sizes."""
    for size in [1, 3, 5]:
        model = SimpleConvNet2d(in_chans=3, out_feats=10, kernel_size=size)
        model[0].apply(lambda x: kernel_size_check(x, size))


def test_fc_net_kwarg_dict_type():
    """Test the types accepted by the `fc_net_kwargs` arg."""
    # Should work with dictionary
    _ = SimpleConvNet2d(out_feats=1, fc_net_kwargs={})

    # Should break with non-dict
    with pytest.raises(TypeError):
        _ = SimpleConvNet2d(out_feats=1, fc_net_kwargs=[1])


def test_in_feats_not_in_dn_kwargs():
    """Test the user cannot supply ``in_feats`` in ``fc_net_kwargs``."""
    with pytest.raises(RuntimeError):
        _ = SimpleConvNet2d(out_feats=1, fc_net_kwargs={"in_feats": 10})


def test_out_feats_not_in_dn_kwargs():
    """Test the user cannot supply ``out_feats`` in ``fc_net_kwargs``."""
    with pytest.raises(RuntimeError):
        _ = SimpleConvNet2d(out_feats=1, fc_net_kwargs={"out_feats": 10})
