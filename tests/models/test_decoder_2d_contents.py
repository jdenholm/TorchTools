"""Tests for the contents of `torch_tools.models._decoder_2d.Decoder_2d`."""

from torch.nn import Conv2d

from torch_tools import Decoder2d
from torch_tools.models._blocks_2d import UpBlock


def test_decoder_2d_number_of_blocks():
    """Test the number of blocks in `Decoder2d`."""
    # Try with 5 blocks
    decoder = Decoder2d(
        in_chans=512,
        out_chans=1,
        num_blocks=5,
        bilinear=False,
        lr_slope=0.123456,
    )

    assert len(decoder) == 5

    # Try with 3 blocks
    decoder = Decoder2d(
        in_chans=256,
        out_chans=1,
        num_blocks=3,
        bilinear=False,
        lr_slope=0.123456,
    )

    assert len(decoder) == 3


def test_decoder_2d_block_types():
    """Test the blocks are the correct types."""
    decoder = Decoder2d(
        in_chans=512,
        out_chans=1,
        num_blocks=5,
        bilinear=False,
        lr_slope=0.123456,
    )

    # All but the last block should be down blocks
    down_blocks = list(decoder.children())[:-1]
    assert all(map(lambda x: isinstance(x, UpBlock), down_blocks))

    # The last block should be a simple Conv2d
    final_conv = list(decoder.children())[-1]
    assert isinstance(final_conv, Conv2d)
