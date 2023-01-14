"""Test the contents of the blocks in `torch_tools.models._blocks_2d`."""
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, Upsample
from torch.nn import MaxPool2d, AvgPool2d, ConvTranspose2d, Sequential

from torch_tools.models._blocks_2d import ConvBlock, DoubleConvBlock
from torch_tools.models._blocks_2d import ResidualBlock, DownBlock
from torch_tools.models._blocks_2d import UpBlock, UNetUpBlock


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

    in_conv = block[0]

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

    out_conv = block[1]

    # Should have three layers in the in_conv block
    assert len(out_conv) == 3, "Wrong number of layers in in conv block."

    assert out_conv[0].in_channels == 321, "Wrong number of input chans."
    assert out_conv[0].out_channels == 321, "Wrong number of output chans."
    assert out_conv[2].negative_slope == 0.123456, "Wrong negative slope."

    assert isinstance(out_conv[0], Conv2d), "Should be Conv2d."
    assert isinstance(out_conv[1], BatchNorm2d), "Should be BatchNorm2d."
    assert isinstance(out_conv[2], LeakyReLU), "Should be LeakyReLU."


def test_residual_block_first_conv_contents():
    """The the contents of the first conv block."""
    block = ResidualBlock(in_chans=123)

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


def test_residual_block_second_conv_contents():
    """The the contents of the second conv block.

    Notes
    -----
    The second conv block should not have a leaky relu at the end. Once
    its output is added to the input, a normal relu is applied.

    """
    block = ResidualBlock(in_chans=123)

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


def test_down_block_contents_pool_assignment():
    """Test pool assignment in `DownBlock`."""
    # Test with max pool
    block = DownBlock(in_chans=123, out_chans=321, pool="max", lr_slope=0.1)
    assert isinstance(block[0], MaxPool2d), "Should be max pool."
    assert block[0].kernel_size == 2, "Kernel size should be 2."
    assert block[0].stride == 2, "Stride should be 2."

    # Test with average pool
    block = DownBlock(in_chans=123, out_chans=321, pool="avg", lr_slope=0.1)
    assert isinstance(block[0], AvgPool2d), "Should be avg pool."
    assert block[0].kernel_size == 2, "Kernel size should be 2."
    assert block[0].stride == 2, "Stride should be 2."


def test_down_block_double_conv_contents():
    """Test the contents of the `DoubleConvBlock` block in `DownBlock`."""
    block = DownBlock(in_chans=123, out_chans=321, pool="max", lr_slope=0.1234)

    in_conv = block[1][0]
    assert isinstance(in_conv[0], Conv2d), "1st layer should be conv"
    assert isinstance(in_conv[1], BatchNorm2d), "2nd layer should be batchnorm"
    assert isinstance(in_conv[2], LeakyReLU), "3rd layer should be leaky relu"

    assert in_conv[0].in_channels == 123
    assert in_conv[0].out_channels == 321
    assert in_conv[1].num_features == 321
    assert in_conv[2].negative_slope == 0.1234

    out_conv = block[1][1]
    assert isinstance(out_conv[0], Conv2d), "Should be conv"
    assert isinstance(out_conv[1], BatchNorm2d), "Should be batchnorm"
    assert isinstance(out_conv[2], LeakyReLU), "Should be leaky relu"

    assert out_conv[0].in_channels == 321
    assert out_conv[0].out_channels == 321
    assert out_conv[1].num_features == 321
    assert out_conv[2].negative_slope == 0.1234


def test_up_block_upsample_contents_with_bilinear_false():
    """Test the contents of upsampling block in `UpBlock`."""
    block = UpBlock(
        in_chans=123,
        out_chans=321,
        bilinear=False,
        lr_slope=0.54321,
    )

    assert isinstance(block[0], ConvTranspose2d), "Should be conv transpose."
    assert block[0].kernel_size == (2, 2), "Kernal size should be 2."
    assert block[0].stride == (2, 2), "Stride should be (2, 2)."
    assert block[0].in_channels == 123, "Should be 123 input channels."
    assert block[0].out_channels == 123, "Should be 123 output channels."


def test_up_block_upsample_contents_with_bilinear_true():
    """Test the contents of upsampling block in `UpBlock`."""
    block = UpBlock(
        in_chans=123,
        out_chans=321,
        bilinear=True,
        lr_slope=0.54321,
    )

    assert isinstance(block[0], Sequential)
    assert len(block[0]) == 2, "Should be two items in the Sequential."

    assert isinstance(block[0][0], Upsample), "First layer should be Upsample."
    assert block[0][0].mode == "bilinear", "Interpolation mode should be bilinear"

    assert isinstance(block[0][1], Conv2d), "Second layer should be Conv2d."

    assert block[0][1].kernel_size == (1, 1), "Kernel size should be (1, 1)."
    assert block[0][1].stride == (1, 1), "Stride should be 1."


def test_up_block_double_conv_first_conv_block_contents():
    """Test contents of first conv block of `DoubleConvBlock` in `UpBlock`."""
    block = UpBlock(
        in_chans=123,
        out_chans=321,
        bilinear=False,
        lr_slope=0.54321,
    )

    assert isinstance(block[1], DoubleConvBlock), "Should be double conv."

    # Test the first conv block of double conv block
    assert isinstance(block[1][0], ConvBlock), "Should be conv block."
    assert isinstance(block[1][0][0], Conv2d), "Should be conv2d."
    assert isinstance(block[1][0][1], BatchNorm2d), "Should be batchnorm."
    assert isinstance(block[1][0][2], LeakyReLU), "Should be leaky relu."

    assert len(block[1][0]) == 3, "Should be three items in the block."

    assert block[1][0][0].in_channels == 123
    assert block[1][0][0].out_channels == 321
    assert block[1][0][1].num_features == 321
    assert block[1][0][2].negative_slope == 0.54321


def test_up_block_double_conv_second_conv_block_contents():
    """Test contents of second conv block of `DoubleConvBlock` in `UpBlock`."""
    block = UpBlock(
        in_chans=123,
        out_chans=321,
        bilinear=False,
        lr_slope=0.54321,
    )

    assert isinstance(block[1], DoubleConvBlock), "Should be double conv."

    # Test the first conv block of double conv block
    assert isinstance(block[1][1], ConvBlock), "Should be conv block."
    assert isinstance(block[1][1][0], Conv2d), "Should be conv2d."
    assert isinstance(block[1][1][1], BatchNorm2d), "Should be batchnorm."
    assert isinstance(block[1][1][2], LeakyReLU), "Should be leaky relu."

    assert len(block[1][1]) == 3, "Should be three items in the block."

    assert block[1][1][0].in_channels == 321
    assert block[1][1][0].out_channels == 321
    assert block[1][1][1].num_features == 321
    assert block[1][1][2].negative_slope == 0.54321


def test_unet_up_block_upsample_contents_with_bilinear_false():
    """Test the contents of the upsample block with bilinear set to false."""
    up_block = UNetUpBlock(
        in_chans=64,
        out_chans=128,
        bilinear=False,
        lr_slope=0.12345,
    )

    assert isinstance(up_block.upsample, ConvTranspose2d)
    assert up_block.upsample.in_channels == 64
    assert up_block.upsample.out_channels == 64 // 2


def test_unet_up_block_upsample_contents_with_bilinear_true():
    """Test the contents of the upsample block with bilinar set to true."""
    up_block = UNetUpBlock(
        in_chans=64,
        out_chans=128,
        bilinear=True,
        lr_slope=0.12345,
    )

    assert isinstance(up_block.upsample, Sequential)
    assert isinstance(up_block.upsample[0], Upsample)
    assert isinstance(up_block.upsample[1], Conv2d)

    assert up_block.upsample[0].mode == "bilinear"
    assert up_block.upsample[1].in_channels == 64
    assert up_block.upsample[1].out_channels == 32


def test_unet_up_block_double_conv_first_conv_block_contents():
    """Test the contents of the first conv block in the double conv block."""
    up_block = UNetUpBlock(
        in_chans=64,
        out_chans=128,
        bilinear=False,
        lr_slope=0.12345,
    )

    double_conv = up_block.double_conv

    # Test the first conv block of the double conv
    assert isinstance(double_conv[0], ConvBlock)
    assert isinstance(double_conv[0][0], Conv2d)
    assert isinstance(double_conv[0][1], BatchNorm2d)
    assert isinstance(double_conv[0][2], LeakyReLU)

    assert double_conv[0][0].in_channels == 64
    assert double_conv[0][0].out_channels == 128
    assert double_conv[0][1].num_features == 128
    assert double_conv[0][2].negative_slope == 0.12345


def test_unet_up_block_double_conv_second_conv_block_contents():
    """Test the contents of the second conv block in the double conv block."""
    up_block = UNetUpBlock(
        in_chans=64,
        out_chans=128,
        bilinear=False,
        lr_slope=0.12345,
    )

    double_conv = up_block.double_conv

    # Test the second conv block of the double conv
    assert isinstance(double_conv[1], ConvBlock)
    assert isinstance(double_conv[1][0], Conv2d)
    assert isinstance(double_conv[1][1], BatchNorm2d)
    assert isinstance(double_conv[1][2], LeakyReLU)

    assert double_conv[1][0].in_channels == 128
    assert double_conv[1][0].out_channels == 128
    assert double_conv[1][1].num_features == 128
    assert double_conv[1][2].negative_slope == 0.12345
