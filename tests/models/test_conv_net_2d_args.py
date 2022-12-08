"""Test arguments accepted by `torch_tools.models._conv_net_2d.ConvNet2d`."""

import pytest

from torch_tools.models import ConvNet2d
from torch_tools.models._encoder_backbones_2d import _encoder_options


def test_out_feats_arg_types():
    """Test the types accepted by the `out_feats` arg."""
    # Should work with ints of one or more
    _ = ConvNet2d(out_feats=1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = ConvNet2d(out_feats=1.0)
    with pytest.raises(TypeError):
        _ = ConvNet2d(out_feats="2")


def test_out_feats_arg_values():
    """Test the values accepted by `out_feats` arg."""
    # Should work with ints of one or more
    _ = ConvNet2d(out_feats=1)
    _ = ConvNet2d(out_feats=1)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = ConvNet2d(out_feats=0)
    with pytest.raises(ValueError):
        _ = ConvNet2d(out_feats=-1)


def test_in_channels_arg_types():
    """Test the types accepted by the `in_channels` arg."""
    # Should work with ints of one or more
    _ = ConvNet2d(out_feats=1, in_channels=1)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = ConvNet2d(out_feats=1, in_channels=1.0)
    with pytest.raises(TypeError):
        _ = ConvNet2d(out_feats=1, in_channels="1")


def test_in_channel_arg_values():
    """Test the values accepted by the `in_channels` arg."""
    # Should work with ints of one or more
    _ = ConvNet2d(out_feats=1, in_channels=1)
    _ = ConvNet2d(out_feats=1, in_channels=2)

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = ConvNet2d(out_feats=1, in_channels=0)
    with pytest.raises(ValueError):
        _ = ConvNet2d(out_feats=1, in_channels=-1)


def test_encoder_style_arg_type():
    """Test the argument types accepted by the `encoder_type` arg."""
    # Should work with allowed strings
    _ = ConvNet2d(out_feats=1, encoder_style="resnet18")

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = ConvNet2d(out_feats=1, encoder_style=123)
    with pytest.raises(TypeError):
        _ = ConvNet2d(out_feats=1, encoder_style=["resnet18"])


def test_encoder_style_accepted_values():
    """Test the values accepted by `encoder_style` arg."""
    # Should work with styles in
    # `torch_tools.models._encoder_backbones._encoder_options`
    for style, _ in _encoder_options.items():
        # Note, we set pretrained to False here to avoid waiting on the
        # pretrained weights to download.
        _ = ConvNet2d(out_feats=1, encoder_style=style, pretrained=False)

    # Should break with strings not in allowed options.
    with pytest.raises(ValueError):
        _ = ConvNet2d(out_feats=1, encoder_style="Elrond of Rivendell")


def test_pretrained_argument_type():
    """Test the types accepted by the `pretrained` arg."""
    # Should work with bool
    _ = ConvNet2d(out_feats=1, pretrained=True)
    _ = ConvNet2d(out_feats=1, pretrained=False)

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = ConvNet2d(out_feats=1, pretrained=1)
    with pytest.raises(TypeError):
        _ = ConvNet2d(out_feats=1, pretrained="True")


def test_pool_style_argument_type():
    """Test the `pool_style` argument type."""
    # Should work with allowed str
    _ = ConvNet2d(out_feats=1, pool_style="avg")

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = ConvNet2d(out_feats=1, pool_style=1)
    with pytest.raises(TypeError):
        _ = ConvNet2d(out_feats=1, pool_style=["Thorin Oakensheild"])


def test_pool_style_argument_values():
    """Test the values accepted by `pool_style` arg."""
    # Should work with the allowed options
    _ = ConvNet2d(out_feats=1, pool_style="avg")
    _ = ConvNet2d(out_feats=1, pool_style="max")
    _ = ConvNet2d(out_feats=1, pool_style="avg-max-concat")

    # Should break with an str which isn't an allowed option
    with pytest.raises(ValueError):
        _ = ConvNet2d(out_feats=1, pool_style="Durin's bane")


def test_dense_net_kwarg_dict_type():
    """Test the types accepted by the `dense_net_kwargs` arg."""
    # Should work with dictionary
    _ = ConvNet2d(out_feats=1, dense_net_kwargs={})

    # Should break with non-dict
    with pytest.raises(TypeError):
        _ = ConvNet2d(out_feats=1, dense_net_kwargs=[1])
