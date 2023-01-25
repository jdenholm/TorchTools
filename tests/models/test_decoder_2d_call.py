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


def test_decoder_2d_output_sizes_at_each_block():
    """Test the output sizes produced by each ``UpBlock``."""
    decoder = Decoder2d(
        in_chans=512,
        out_chans=111,
        num_blocks=5,
        bilinear=False,
        lr_slope=0.123456,
    )

    chans = 512
    height, width = 8, 8
    batch = rand(10, chans, height, width)

    for block in list(decoder.children())[:-1]:

        batch = block(batch)

        chans //= 2
        height *= 2
        width *= 2

        assert batch.shape == (10, chans, height, width)
