"""Tests for the contents of the blocks in `torch_tools.models._blocks_2d`"""
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU

from torch_tools.models._blocks_2d import ConvBlock, DoubleConvBlock, ResBlock

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
    assert block[0].in_channels == 123, "Wrong input channels."
    assert block[0].out_channels == 321, "Wrong output channels."

    # Check the right number of layers are in the block
    assert len(block) == 3, "Wrong number of layers in the block."

    # Check the layers are in the right order
    assert isinstance(block[0], Conv2d), "1st layer should be conv 2d."
    assert isinstance(block[1], BatchNorm2d), "2nd layer should be bnorm."
    assert isinstance(block[2], LeakyReLU), "3rd layer should be leaky relu."

    # Check the batchnorm has the right number of features
    assert block[1].num_features == 321, "Wrong num feats in batchnorm."

    # Check the leaky relu's negative slope is set correctly
    assert block[2].negative_slope == 0.12345, "Leaky ReLU slope wrong."


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
    assert block[0].in_channels == 123, "Wrong input channels."
    assert block[0].out_channels == 321, "Wrong output channels."

    # Check the right number of layers are in the block
    assert len(block) == 2, "Wrong number of layers in the block."

    # Check the layers are in the right order
    assert isinstance(block[0], Conv2d), "1st layer should be conv 2d."
    assert isinstance(block[1], BatchNorm2d), "2nd layer should be bnorm."

    # Check the batchnorm has the right number of features
    assert block[1].num_features == 321, "Wrong num feats in batchnorm."


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
    assert block[0].in_channels == 123, "Wrong input channels."
    assert block[0].out_channels == 321, "Wrong output channels."

    # Check the right number of layers are in the block
    assert len(block) == 2, "Wrong number of layers in the block."

    # Check the layers are in the right order
    assert isinstance(block[0], Conv2d), "1st layer should be conv 2d."
    assert isinstance(block[1], LeakyReLU), "3rd layer should be leaky relu."

    # Check the leaky relu's negative slope is set correctly
    assert block[1].negative_slope == 0.12345, "Leaky ReLU slope wrong."


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
    assert block[0].in_channels == 123, "Wrong input channels."
    assert block[0].out_channels == 321, "Wrong output channels."

    # Check the right number of layers are in the block
    assert len(block) == 1, "Wrong number of layers in the block."

    # Check the layers are in the right order
    assert isinstance(block[0], Conv2d), "1st layer should be conv 2d."


def test_double_conv_block_in_conv_contents():
    """Test the contents of `DoubleConvBlock`'s `in_conv` are as expected."""
    block = DoubleConvBlock(in_chans=123, out_chans=321, lr_slope=0.123456)

    in_conv = block.in_conv

    # Should have three layers in the in_conv block
    assert len(in_conv) == 3, "Wrong number of layers in in conv block."

    assert in_conv[0].in_channels == 123, "Wrong number of input chans."
    assert in_conv[0].out_channels == 321, "Wrong number of output chans."
    assert in_conv[2].negative_slope == 0.123456, "Wrong negative slope."

    assert isinstance(in_conv[0], Conv2d), "Should be Conv2d."
    assert isinstance(in_conv[1], BatchNorm2d), "Should be BatchNorm2d."
    assert isinstance(in_conv[2], LeakyReLU), "Should be LeakyReLU."


def test_double_conv_block_out_conv_contents():
    """Test the contents of `DoubleConvBlock`'s `out_conv` are as expected."""
    block = DoubleConvBlock(in_chans=123, out_chans=321, lr_slope=0.123456)

    out_conv = block.out_conv

    # Should have three layers in the in_conv block
    assert len(out_conv) == 3, "Wrong number of layers in in conv block."

    assert out_conv[0].in_channels == 321, "Wrong number of input chans."
    assert out_conv[0].out_channels == 321, "Wrong number of output chans."
    assert out_conv[2].negative_slope == 0.123456, "Wrong negative slope."

    assert isinstance(out_conv[0], Conv2d), "Should be Conv2d."
    assert isinstance(out_conv[1], BatchNorm2d), "Should be BatchNorm2d."
    assert isinstance(out_conv[2], LeakyReLU), "Should be LeakyReLU."


def test_res_block_first_conv_contents():
    """The the contents of the first conv block."""
    block = ResBlock(in_chans=123)

    first_conv = block.first_conv

    assert isinstance(first_conv, ConvBlock)

    assert len(first_conv) == 3, "There should be 3 layers in the block."

    msg = "1st layer should be conv."
    assert isinstance(first_conv[0], Conv2d), msg

    msg = "2nd layer should be batchnorm."
    assert isinstance(first_conv[1], BatchNorm2d), msg

    msg = "3rd layer should be leaky relu."
    assert isinstance(first_conv[2], LeakyReLU), msg

    msg = "Conv should have 123 input chans."
    assert first_conv[0].in_channels == 123, msg

    msg = "Conv should have 123 output chans."
    assert first_conv[0].out_channels == 123, msg

    msg = "Batchnorm should have 123 features."
    assert first_conv[1].num_features == 123, msg

    msg = "Leaky relu should have negative slope of 0.0."
    assert first_conv[2].negative_slope == 0.0, msg


def test_res_block_second_conv_contents():
    """The the contents of the second conv block.

    Notes
    -----
    The second conv block should not have a leaky relu at the end. Once
    its output is added to the input, a normal relu is applied.

    """
    block = ResBlock(in_chans=123)

    second_conv = block.second_conv

    assert isinstance(second_conv, ConvBlock)

    assert len(second_conv) == 2, "There should be 2 layers in the block."

    msg = "1st layer should be conv."
    assert isinstance(second_conv[0], Conv2d), msg

    msg = "2nd layer should be batchnorm."
    assert isinstance(second_conv[1], BatchNorm2d), msg

    msg = "Conv should have 123 input chans."
    assert second_conv[0].in_channels == 123, msg

    msg = "Conv should have 123 output chans."
    assert second_conv[0].out_channels == 123, msg

    msg = "Batchnorm should have 123 features."
    assert second_conv[1].num_features == 123, msg
