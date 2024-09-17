"""Two-dimensional convolutional blocks."""

from typing import List

from torch import Tensor, cat  # pylint: disable=no-name-in-module
from torch.nn import Module, Conv2d, BatchNorm2d, LeakyReLU, Sequential, ReLU
from torch.nn import ConvTranspose2d, Upsample
from torch.nn import MaxPool2d, AvgPool2d, Dropout2d

from torch.nn.functional import pad


from torch_tools.models._argument_processing import (
    process_num_feats,
    process_2d_kernel_size,
    process_negative_slope_arg,
    process_boolean_arg,
    process_str_arg,
    process_2d_block_style_arg,
    process_dropout_prob,
)

from torch_tools.torch_utils import disable_biases

# pylint: disable=too-many-arguments


class ConvBlock(Sequential):
    """Single 2D convolutional block.

    Parameters
    ----------
    in_chans : int
        The number of input channels the block should take.
    out_chans : int
        The number of output channels the block should produce.
    kernel_size : int, optional
        The kernel size to use in the ``Conv2d`` layers. Should be odd and
        positive.
    batch_norm : bool
        Should we include a ``BatchNorm2d`` layer?
    leaky_relu : bool
        Should we include a ``LeakyReLU`` layer?
    lr_slope : float, optional
        The negative slope to use in the ``LeakyReLU`` (use 0.0 for ``ReLU``).
    dropout : float, optional
        The dropout to apply at the output of the block.

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        batch_norm: bool = True,
        leaky_relu: bool = True,
        lr_slope: float = 0.1,
        dropout: float = 0.0,
    ):
        """Build `SingleConvBlock`."""
        super().__init__(
            *self._layers(
                process_num_feats(in_chans),
                process_num_feats(out_chans),
                process_2d_kernel_size(kernel_size),
                process_boolean_arg(batch_norm),
                process_boolean_arg(leaky_relu),
                process_negative_slope_arg(lr_slope),
                process_dropout_prob(dropout),
            )
        )

    @staticmethod
    def _layers(
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        batch_norm: bool,
        leaky_relu: bool,
        lr_slope: float,
        dropout: float,
    ) -> List[Module]:
        """Stack the block's layers in a `Sequential`.

        Parameters
        ----------
        (See class docstring).

        Returns
        -------
        List[Module]
            The block's layers in a list.

        """
        layers: List[Module]
        layers = [
            Conv2d(
                in_chans,
                out_chans,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
            )
        ]

        if batch_norm is True:
            layers.append(BatchNorm2d(out_chans))

        if leaky_relu is True:
            layers.append(LeakyReLU(lr_slope))

        if dropout != 0.0:
            layers.append(Dropout2d(p=dropout))

        return layers


class DoubleConvBlock(Sequential):
    """Double convolutional block.

    Parameters
    ----------
    in_chans : int
        The number of input channels the block should take.
    out_chans : int
        The number of output channels the block should produce.
    lr_slope : float, optional
        The slope to use in the `LeakyReLU` layers.
    kernel_size : int
        The size of the kernel to use in the ``ConvBlock``s. Should be odd,
        positive integers.

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        lr_slope: float,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        """Build `DoubleConvBlock`."""
        super().__init__(
            ConvBlock(
                process_num_feats(in_chans),
                process_num_feats(out_chans),
                batch_norm=True,
                leaky_relu=True,
                lr_slope=process_negative_slope_arg(lr_slope),
                kernel_size=process_2d_kernel_size(kernel_size),
                dropout=0.0,
            ),
            ConvBlock(
                process_num_feats(out_chans),
                process_num_feats(out_chans),
                batch_norm=True,
                leaky_relu=True,
                lr_slope=process_negative_slope_arg(lr_slope),
                kernel_size=process_2d_kernel_size(kernel_size),
                dropout=process_dropout_prob(dropout),
            ),
        )


class ResidualBlock(Module):
    """Residual block.

    Parameters
    ----------
    in_chans : int
        The number of input channels.
    kernel_size : int
        Size of the square convolutional kernel to use in the ``Conv2d``
        layers. Must be an odd, positive, int.

    """

    def __init__(self, in_chans: int, kernel_size: int = 3):
        """Build `ResidualBlock`."""
        super().__init__()
        self.first_conv = ConvBlock(
            in_chans,
            in_chans,
            kernel_size=process_2d_kernel_size(kernel_size),
            batch_norm=True,
            leaky_relu=True,
            lr_slope=0.0,
            dropout=0.0,
        )
        self.second_conv = ConvBlock(
            in_chans,
            in_chans,
            kernel_size=process_2d_kernel_size(kernel_size),
            batch_norm=True,
            leaky_relu=False,
            dropout=0.0,
        )

        self.relu = ReLU()

        self.apply(disable_biases)

    def forward(self, batch: Tensor) -> Tensor:
        """Pass `batch` through the block.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        Returns
        -------
        Tensor
            The result of passing `batch` through the block.

        """
        identity = batch
        out = self.first_conv(batch)
        out = self.second_conv(out)
        out += identity
        return self.relu(out)


class ConvResBlock(Sequential):
    """A single ``ConvBlock`` followed by a ``ResidualBlock``.

    Parameters
    ----------
    in_chans : int
        The number of input channels.
    out_chans : int
        The number of output channels.
    lr_slope : float
        The negative slope to use in the leaky relu layers.
    kernel_size : int
        Size of the square convolutional kernel to use.
    dropout : float, optional
        The dropout probability to apply at the output of
        the block.

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        lr_slope: float,
        kernel_size: int = 3,
        dropout: float = 0.123,
    ):
        """Build ``ConvResBlock``."""
        super().__init__(
            *self._layers(
                in_chans,
                out_chans,
                lr_slope,
                kernel_size,
                dropout,
            )
        )

    @staticmethod
    def _layers(
        in_chans: int,
        out_chans: int,
        lr_slope: float,
        kernel_size: int,
        dropout: float,
    ) -> List[Module]:
        """List the layers in the block.

        Parameters
        ----------
        See class-level docstring.

        Returns
        -------
        layers : List[Module]
            A list of the layers in the model.

        """
        layers = [
            ConvBlock(
                process_num_feats(in_chans),
                process_num_feats(out_chans),
                kernel_size=process_2d_kernel_size(kernel_size),
                batch_norm=process_boolean_arg(True),
                leaky_relu=process_boolean_arg(True),
                lr_slope=process_negative_slope_arg(lr_slope),
                dropout=0.0,
            ),
            ResidualBlock(
                process_num_feats(out_chans),
                kernel_size=process_2d_kernel_size(kernel_size),
            ),
        ]

        if process_dropout_prob(dropout) != 0.0:
            layers.append(Dropout2d(p=process_dropout_prob(dropout)))

        return layers


class DownBlock(Sequential):
    """Down-sampling block which reduces image size by a factor of 2.

    Parameters
    ----------
    in_chans : int
        The number of input channels the block should take.
    out_chans : int
        The number of output channels the block should take.
    pool : str
        The style of the pooling layer to use: can be `"avg"` or `"max"`.
    lr_slope : float
        The negative slope to use in the ``LeakyReLU`` arguments.
    kernel_size : int, optional
        The size of the square convolutional kernel to use on the ``Conv2d``
        layers. Must be an odd, positive, int.
    block_style : str, optional
        Encoding block style. See
        ``torch_tools.models._blocks_2d._conv_blocks`` for options.
    dropout : float, optional
        The dropout probablity to apply at the block's output.

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        pool: str,
        lr_slope: float,
        kernel_size: int = 3,
        block_style: str = "double_conv",
        dropout: float = 0.0,
    ):
        """Build `DownBlock`."""
        super().__init__(
            self._pools[process_str_arg(pool).lower()](
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            _conv_blocks[process_2d_block_style_arg(block_style)](
                in_chans=process_num_feats(in_chans),
                out_chans=process_num_feats(out_chans),
                lr_slope=process_negative_slope_arg(lr_slope),
                kernel_size=process_2d_kernel_size(kernel_size),
                dropout=process_dropout_prob(dropout),
            ),
        )

    _pools = {"max": MaxPool2d, "avg": AvgPool2d}


class UpBlock(Sequential):
    """Upsampling block which increases image size by a factor of 2.

    Parameters
    ----------
    in_chans : int
        The number of inpuuts channels the block should take.
    out_chans : int
        The number of output channels the block should take.
    bilinear : bool
        Determines whether the block uses bilinear interpolation (``True``) or
        ``ConvTranspose2d`` (``False``).
    lr_slope : float
        Negative slope to use in the ``LeakyReLU`` layers.
    block_style : str, optional
        Encoding block style. See
        ``torch_tools.models._blocks_2d._conv_blocks`` for options.
    kernel_size : int, optional
        Size of the covolutional kernel. Must be an odd, positive, int.

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        bilinear: bool,
        lr_slope: float,
        block_style: str = "double_conv",
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        """Build `UpBlock`."""
        super().__init__(
            self._get_upsampler(
                process_num_feats(in_chans),
                process_boolean_arg(bilinear),
            ),
            _conv_blocks[block_style](
                process_num_feats(in_chans),
                process_num_feats(out_chans),
                lr_slope=process_negative_slope_arg(lr_slope),
                kernel_size=process_2d_kernel_size(kernel_size),
                dropout=process_dropout_prob(dropout),
            ),
        )

    @staticmethod
    def _get_upsampler(in_chans: int, bilinear: bool) -> Module:
        """Return the upsampling component of the block.

        Parameters
        ----------
        in_chans : int
            The number of input channels.
        bilinear : bool
            Bool controlling upsampling method. If `True`, we use bilinear
            interpolation. If `False`, we use `ConvTranspose2d`.

        Returns
        -------
        upsample : Module
            The upsampling component of the block.

        """
        upsample: Module
        if bilinear is True:
            upsample = Sequential(
                Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                Conv2d(in_chans, in_chans, kernel_size=1, stride=1),
            )
        else:
            upsample = ConvTranspose2d(
                in_chans,
                in_chans,
                kernel_size=2,
                stride=2,
            )
        return upsample


class UNetUpBlock(Module):
    """Upsampling block to be used in the second half of a UNet.

    Parameters
    ----------
    in_chans : int
        The number of input channels.
    out_chans : int
        The number of output channels.
    bilinear : bool
        If ``True``, the upsample is done using bilinear interpolation using
        ``torch.nn.Upsample``. Otherwise we use a ``ConvTranspose2d``.
    lr_slope : float
        The negative slope to use in the ``LeakyReLU``.
    kernel_size : int
        The size of the square convolutional kernel to use in the convolutional
        layers, Should be an odd, positive, int.
    block_style : str
        Style of the convolutional block: ``"double_conv"`` or ``"conv_res"``.
    dropout : float, optional
        The dropout probability to apply at the block's output.

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        bilinear: bool,
        lr_slope: float,
        kernel_size: int = 3,
        block_style: str = "double_conv",
        dropout: float = 0.0,
    ):
        """Build `UNetUpBlock`."""
        super().__init__()
        self._in_chans = self._process_in_chans(in_chans)
        self._out_chans = process_num_feats(out_chans)

        self.upsample = self._get_upsampler(process_boolean_arg(bilinear))

        if process_2d_block_style_arg(block_style) == "double_conv":
            self.conv_block: Module = DoubleConvBlock(
                self._in_chans,
                self._out_chans,
                lr_slope=lr_slope,
                kernel_size=process_2d_kernel_size(kernel_size),
                dropout=process_dropout_prob(dropout),
            )
        else:
            self.conv_block = ConvResBlock(
                self._in_chans,
                self._out_chans,
                lr_slope,
                kernel_size,
                dropout=process_dropout_prob(dropout),
            )

    @staticmethod
    def _process_in_chans(in_chans: int) -> int:
        """Process `in_chans` arg.

        Parameters
        ----------
        in_chans : int
            The number of input channels requested by the user.

        Returns
        -------
        in_chans : int
            The number of input channels requested by the user.

        Raises
        ------
        TypeError
            If `in_chans` is not an int.
        ValueError
            If `in_chans` is less than 2.
        ValueError
            If `in_chans` is not even.

        """
        if not isinstance(in_chans, int):
            raise TypeError(f"in_chans should be int. Got {type(in_chans)}.")
        if in_chans < 2:
            raise ValueError(f"in_chans should be 2 or more. Got {in_chans}.")
        if (in_chans % 2) != 0:
            raise ValueError(f"in_chans should be even. Got {in_chans}.")

        return in_chans

    def _get_upsampler(self, bilinear: bool) -> Module:
        """Return the upsampling layer.

        Parameters
        ----------
        bilinear : bool
            Whether to use bilinear interpolation to upsample (`True`) or
            a `ConvTranspose2d` (`False`).

        Returns
        -------
        Module
            An upsampling block which increases the input dimensionality by a
            factor of 2.

        """
        if bilinear is True:
            return Sequential(
                Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                Conv2d(self._in_chans, self._in_chans // 2, kernel_size=1),
            )
        return ConvTranspose2d(
            self._in_chans,
            self._in_chans // 2,
            kernel_size=2,
            stride=2,
        )

    @staticmethod
    def _channel_size_check(to_upsample: Tensor, down_features: Tensor):
        """Check the channel sizes are offset by a factor of 2.

        Parameters
        ----------
        to_upsample : Tensor
            The image batch to be upsampled.
        down_features : Tensor
            The batch from the UNet's down path.

        Raises
        ------
        RuntimeError
            If the number of channels in `to_upsample` is not two times
            greater than the number of channels in `down_features`.

        """
        up_chans = to_upsample.shape[1]
        down_chans = down_features.shape[1]
        if not (up_chans / down_chans) == 2:
            msg = "Channel sizes should be off by a factor of 2. "
            msg += f"Got {up_chans} and {down_chans}"
            raise RuntimeError(msg)

    def _to_upsample_channel_check(self, to_upsample: Tensor):
        """Check the number of channels in `to_upsample` match `_in_chans`.

        Parameters
        ----------
        to_upsample : Tensor
            The Tensor to be upsampled.

        Raises
        ------
        RuntimeError
            If `to_upsample` has the wrong number of channels.

        """
        if not to_upsample.shape[1] == self._in_chans:
            msg = f"to_upsample should have {self._in_chans} channels. "
            msg += f"Got {to_upsample.shape[1]}."
            raise RuntimeError(msg)

    def _input_size_check(self, to_upsample: Tensor, down_features: Tensor):
        """Make sure `to_upsample` is smaller than `down_features`.

        Parameters
        ----------
        to_upsample : Tensor
            The features to be upsampled.
        down_features : Tensor
            The features from the down path to be concatenated with the
            upsampled features.

        Raises
        ------
        RuntimeError
            If the height and width of `to_upsample` is greater than or equal
            to `down_features`, something has gone wrong.

        """
        _, _, to_upsample_height, to_upsample_width = to_upsample.shape
        _, _, down_features_height, down_features_width = down_features.shape

        up_height_bigger = to_upsample_height >= down_features_height
        up_width_bigger = to_upsample_width >= down_features_width

        if up_height_bigger or up_width_bigger:
            msg = "The features to be upsampled have a height and width of "
            msg += f"{to_upsample_height} and {to_upsample_width}, and the "
            msg += "features to be downsampled have a height and width of "
            msg += f"{down_features_height} and {down_features_width}. "
            msg += "The features to be upsampled should be smaller in "
            msg += "each dimension in a UNet."
            raise RuntimeError(msg)

    def forward(self, to_upsample: Tensor, down_features: Tensor) -> Tensor:
        """Unet skip-connection forward step.

        Parameters
        ----------
        to_upsample : Tensor
            The batch to be upsampled by the layer.
        down_features : Tensor
            The corresponding down features to be concatenated with the
            upsampled `to_upsample`.

        Returns
        -------
        Tensor
            The output of the UNet upsampling skip connection.

        """
        self._channel_size_check(to_upsample, down_features)
        self._to_upsample_channel_check(to_upsample)
        self._input_size_check(to_upsample, down_features)

        upsampled = self.upsample(to_upsample)

        height_diff = down_features.shape[2] - upsampled.shape[2]
        width_diff = down_features.shape[3] - upsampled.shape[3]

        padding = (
            width_diff // 2,  # Left padding
            width_diff - width_diff // 2,  # Right padding
            height_diff // 2,  # Top padding
            height_diff - height_diff // 2,  # Bottom padding
        )

        upsampled = pad(upsampled, padding)  # pylint: disable=not-callable

        # Concatenate along the channel dimension (dim=1) (N, C, H, W)
        concatenated = cat([down_features, upsampled], dim=1)
        return self.conv_block(concatenated)


_conv_blocks = {"double_conv": DoubleConvBlock, "conv_res": ConvResBlock}
