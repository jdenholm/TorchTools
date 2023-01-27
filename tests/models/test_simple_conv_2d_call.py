"""Test the call behaviour of ``torch_tools.SimpleConvNet2d``."""
from itertools import product

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


def test_number_of_output_features():
    """Test the number of output features."""
    model = SimpleConvNet2d(in_chans=3, out_feats=123)
    assert model(rand(10, 3, 50, 50)).shape == (10, 123)

    model = SimpleConvNet2d(in_chans=3, out_feats=321)
    assert model(rand(10, 3, 50, 50)).shape == (10, 321)

    model = SimpleConvNet2d(in_chans=3, out_feats=521)
    assert model(rand(10, 3, 50, 50)).shape == (10, 521)


def test_call_with_different_pool_types():
    """Test with every pool type."""
    down_pools = ["max", "avg"]
    adaptive_pools = ["max", "avg", "avg-max-concat"]

    for down, adaptive in product(down_pools, adaptive_pools):
        model = SimpleConvNet2d(
            in_chans=3,
            out_feats=123,
            adaptive_pool=adaptive,
            downsample_pool=down,
        )
        _ = model(rand(10, 3, 50, 50))
