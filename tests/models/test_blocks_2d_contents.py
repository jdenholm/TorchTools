"""Tests for the contents of the blocks in `torch_tools.models._blocks_2d`"""
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU

from torch_tools.models._blocks_2d import ConvBlock, DoubleConvBlock

# pylint: disable=protected-access


def test_conv_block_contents_with_batchnorm_and_leaky_relu():
    """Test the contents with batchnorm and leaky relu layers included."""
    block = ConvBlock(
        in_chans=123,
        out_chans=321,
        batch_norm=True,
        leaky_relu=True,
        lr_slope=0.12345,
    )

    # Check the channels are set correctly
    assert block._fwd[0].in_channels == 123, "Wrong input channels."
    assert block._fwd[0].out_channels == 321, "Wrong output channels."

    # Check the right number of layers are in the block
    assert len(block._fwd) == 3, "Wrong number of layers in the block."

    # Check the layers are in the right order
    assert isinstance(block._fwd[0], Conv2d), "1st layer should be conv 2d."
    assert isinstance(block._fwd[1], BatchNorm2d), "2nd layer should be bnorm."
    assert isinstance(block._fwd[2], LeakyReLU), "3rd layer should be leaky relu."

    # Check the batchnorm has the right number of features
    assert block._fwd[1].num_features == 321, "Wrong num feats in batchnorm."

    # Check the leaky relu's negative slope is set correctly
    assert block._fwd[2].negative_slope == 0.12345, "Leaky ReLU slope wrong."


def test_conv_block_contents_with_batchnorm_and_no_leaky_relu():
    """Test the contents of `ConvBlock` with a batchnorm and no `leakReLU`."""
    block = ConvBlock(
        in_chans=123,
        out_chans=321,
        batch_norm=True,
        leaky_relu=False,
        lr_slope=0.12345,
    )

    # Check the channels are set correctly
    assert block._fwd[0].in_channels == 123, "Wrong input channels."
    assert block._fwd[0].out_channels == 321, "Wrong output channels."

    # Check the right number of layers are in the block
    assert len(block._fwd) == 2, "Wrong number of layers in the block."

    # Check the layers are in the right order
    assert isinstance(block._fwd[0], Conv2d), "1st layer should be conv 2d."
    assert isinstance(block._fwd[1], BatchNorm2d), "2nd layer should be bnorm."

    # Check the batchnorm has the right number of features
    assert block._fwd[1].num_features == 321, "Wrong num feats in batchnorm."


def test_conv_block_contents_with_no_batchnorm_and_with_leaky_relu():
    """Test the contents of `ConvBlock` with no batchnorm and a leaky relu."""
    block = ConvBlock(
        in_chans=123,
        out_chans=321,
        batch_norm=False,
        leaky_relu=True,
        lr_slope=0.12345,
    )

    # Check the channels are set correctly
    assert block._fwd[0].in_channels == 123, "Wrong input channels."
    assert block._fwd[0].out_channels == 321, "Wrong output channels."

    # Check the right number of layers are in the block
    assert len(block._fwd) == 2, "Wrong number of layers in the block."

    # Check the layers are in the right order
    assert isinstance(block._fwd[0], Conv2d), "1st layer should be conv 2d."
    assert isinstance(block._fwd[1], LeakyReLU), "3rd layer should be leaky relu."

    # Check the leaky relu's negative slope is set correctly
    assert block._fwd[1].negative_slope == 0.12345, "Leaky ReLU slope wrong."


def test_conv_block_contents_with_no_batchnorm_or_leaky_relu():
    """Test the contents of `ConvBlock` with no batchnorm or leaky relu."""
    block = ConvBlock(
        in_chans=123,
        out_chans=321,
        batch_norm=False,
        leaky_relu=False,
        lr_slope=0.12345,
    )

    # Check the channels are set correctly
    assert block._fwd[0].in_channels == 123, "Wrong input channels."
    assert block._fwd[0].out_channels == 321, "Wrong output channels."

    # Check the right number of layers are in the block
    assert len(block._fwd) == 1, "Wrong number of layers in the block."

    # Check the layers are in the right order
    assert isinstance(block._fwd[0], Conv2d), "1st layer should be conv 2d."


def test_double_conv_block_in_conv_contents():
    """Test the contents of `DoubleConvBlock`'s `in_conv` are as expected."""
    block = DoubleConvBlock(in_chans=123, out_chans=321, lr_slope=0.123456)

    in_conv = block._in_conv

    # Should have three layers in the in_conv block
    assert len(in_conv._fwd) == 3, "Wrong number of layers in in conv block."

    assert in_conv._fwd[0].in_channels == 123, "Wrong number of input chans."
    assert in_conv._fwd[0].out_channels == 321, "Wrong number of output chans."
    assert in_conv._fwd[2].negative_slope == 0.123456, "Wrong negative slope."

    assert isinstance(in_conv._fwd[0], Conv2d), "Should be Conv2d."
    assert isinstance(in_conv._fwd[1], BatchNorm2d), "Should be BatchNorm2d."
    assert isinstance(in_conv._fwd[2], LeakyReLU), "Should be LeakyReLU."


def test_double_conv_block_out_conv_contents():
    """Test the contents of `DoubleConvBlock`'s `out_conv` are as expected."""
    block = DoubleConvBlock(in_chans=123, out_chans=321, lr_slope=0.123456)

    out_conv = block._out_conv

    # Should have three layers in the in_conv block
    assert len(out_conv._fwd) == 3, "Wrong number of layers in in conv block."

    assert out_conv._fwd[0].in_channels == 321, "Wrong number of input chans."
    assert out_conv._fwd[0].out_channels == 321, "Wrong number of output chans."
    assert out_conv._fwd[2].negative_slope == 0.123456, "Wrong negative slope."

    assert isinstance(out_conv._fwd[0], Conv2d), "Should be Conv2d."
    assert isinstance(out_conv._fwd[1], BatchNorm2d), "Should be BatchNorm2d."
    assert isinstance(out_conv._fwd[2], LeakyReLU), "Should be LeakyReLU."
