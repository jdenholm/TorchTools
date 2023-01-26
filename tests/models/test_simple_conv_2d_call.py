"""Test the call behaviour of ``torch_tools.SimpleConvNet2d``."""
from torch import rand  # pylint: disable=no-name-in-module

from torch_tools import SimpleConvNet2d


def test_accepted_input_channels():
    """Test the number of inputs channels accepted."""
    model = SimpleConvNet2d(in_chans=123, out_feats=8)
    _ = model(rand(10, 123, 50, 50))

    model = SimpleConvNet2d(in_chans=321, out_feats=8)
    _ = model(rand(10, 321, 50, 50))

    model = SimpleConvNet2d(in_chans=666, out_feats=8)
    _ = model(rand(10, 666, 50, 50))
