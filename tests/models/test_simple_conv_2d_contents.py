"""Test the contents of ``torch_tools.SimpleConvNet2d``."""

from torch.nn import AdaptiveAvgPool2d, Flatten, Linear

from torch_tools import SimpleConvNet2d, Encoder2d


def test_simple_conv_net_contents():
    """Test the contents of ``SimpleConvNet2d``."""
    model = SimpleConvNet2d(in_chans=123, out_feats=8, adaptive_pool="avg")

    assert len(model) == 4
    assert isinstance(model[0], Encoder2d)
    assert isinstance(model[1], AdaptiveAvgPool2d)
    assert isinstance(model[2], Flatten)
    assert isinstance(model[3], Linear)
