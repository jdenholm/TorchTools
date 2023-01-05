"""Tests for the call methods of blocks in `torch_tools.models._blocks_2d`."""

from torch import rand  # pylint: disable=no-name-in-module

from torch_tools.models._blocks_2d import ConvBlock


def test_conv_block_call_return_shapes_with_batchnorm_and_leaky_relu():
    """Test the return shapes produced by `ConvBlock` are correct.

    Notes
    -----
    Test the block returns the correct shape when we include both a
    batchnorm and leaky relu layer.


    """
    block = ConvBlock(
        in_chans=123,
        out_chans=321,
        batch_norm=True,
        leaky_relu=True,
    )
    assert block(rand(10, 123, 50, 100)).shape == (10, 321, 50, 100)


def test_conv_block_call_return_shapes_with_batchnorm_and_no_leaky_relu():
    """Test the return shapes produced by `ConvBlock` are correct.

    Notes
    -----
    Test the block returns the correct shape when we only include a batchnorm
    and do not include a leaky relu.

    """
    block = ConvBlock(
        in_chans=111,
        out_chans=222,
        batch_norm=True,
        leaky_relu=False,
    )
    assert block(rand(10, 111, 12, 21)).shape == (10, 222, 12, 21)

def test_conv_block_call_return_shapes_with_no_batchnorm_and_no_leaky_relu():
    """Test the return shapes produced by `ConvBlock` are correct.

    Notes
    -----
    Test the block returns the correct shape when we don't include a batchnorm
    or a leaky relu.

    """
    block = ConvBlock(
        in_chans=1,
        out_chans=321,
        batch_norm=False,
        leaky_relu=False,
    )
    assert block(rand(10, 1, 50, 50)).shape == (10, 321, 50, 50)
