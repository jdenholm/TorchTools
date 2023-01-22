"""Test the call behaviour in `Decoder2d` is as expected."""

from torch import rand  # pylint: disable=no-name-in-module

from torch_tools.models import Decoder2d


def test_decoder_2d_returns_images_of_the_right_shape():
    """Test the dimensionality of the images returned by `Decoder2d`."""
    decoder = Decoder2d(
        in_chans=512,
        out_chans=111,
        num_blocks=5,
        bilinear=False,
        lr_slope=0.123456,
    )

    mini_batch = rand(10, 512, 8, 4)
    assert decoder(mini_batch).shape == (10, 111, 128, 64)

    mini_batch = rand(10, 512, 4, 8)
    assert decoder(mini_batch).shape == (10, 111, 64, 128)
