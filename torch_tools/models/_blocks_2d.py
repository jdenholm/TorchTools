"""Two-dimensional convolutional blocks."""
from typing import List

from torch import Tensor, cat  # pylint: disable=no-name-in-module
from torch.nn import Module, Conv2d, BatchNorm2d, LeakyReLU, Sequential, ReLU
from torch.nn import ConvTranspose2d, Upsample
from torch.nn import MaxPool2d, AvgPool2d

from torch.nn.functional import pad


from torch_tools.models._argument_processing import (
    process_num_feats,
    process_negative_slope_arg,
    process_boolean_arg,
)


class ConvBlock(Module):
    """Single 2D convolutional block.

    Parameters
    ----------
    in_chans : int
        The number of input channels the block should take.
    out_chans : int
        The number of output channels the block should produce.
    batch_norm : bool
        Should we include a `BatchNorm2d` layer?
    leaky_relu : bool
        Should we include a `LeakyReLU` layer?
    lr_slope : float,, optional
        The negative slope to use in the `LeakyReLU` (use 0.0 for `ReLU`).

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        batch_norm: bool = True,
        leaky_relu: bool = True,
        lr_slope: float = 0.1,
    ):
        """Build `SingleConvBlock`."""
        super().__init__()
        self._fwd = self._layers(
            process_num_feats(in_chans),
            process_num_feats(out_chans),
            process_boolean_arg(batch_norm),
            process_boolean_arg(leaky_relu),
            process_negative_slope_arg(lr_slope),
        )

    @staticmethod
    def _layers(
        in_chans: int,
        out_chans: int,
        batch_norm: bool,
        leaky_relu: bool,
        lr_slope: float,
    ) -> Sequential:
        """Stack the block's layers in a `Sequential`.

        Parameters
        ----------
        (See class docstring).

        Returns
        -------
        Sequential
            The block's layers in a `Sequential`.

        """
        layers: List[Module]
        layers = [Conv2d(in_chans, out_chans, kernel_size=3, padding=1)]

        if batch_norm is True:
            layers.append(BatchNorm2d(out_chans))

        if leaky_relu is True:
            layers.append(LeakyReLU(lr_slope))

        return Sequential(*layers)

    def forward(self, batch: Tensor) -> Tensor:
        """Pass `batch` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        Returns
        -------
        Tensor
            The result of passing `batch` through the model.

        """
        return self._fwd(batch)


class DoubleConvBlock(Module):
    """Double convolutional block.

    Parameters
    ----------
    in_chans : int
        The number of input channels the block should take.
    out_chans : int
        The number of output channels the block should produce.
    lr_slope : float, optional
        The slope to use in the `LeakyReLU` layers.

    """

    def __init__(self, in_chans: int, out_chans: int, lr_slope: float = 0.1):
        """Build `DoubleConvBlock`.

        Parameters
        ----------
        in_chans : int
            The number of input channels the block should take.
        out_chans : int
            The number of output channels the block should take.
        lr_slope : float, optional
            The negative slope to use in the `LeakyReLU`.

        """
        super().__init__()
        self.conv1 = ConvBlock(
            process_num_feats(in_chans),
            process_num_feats(out_chans),
            batch_norm=True,
            leaky_relu=True,
            lr_slope=process_negative_slope_arg(lr_slope),
        )
        self.conv2 = ConvBlock(
            process_num_feats(out_chans),
            process_num_feats(out_chans),
            batch_norm=True,
            leaky_relu=True,
            lr_slope=process_negative_slope_arg(lr_slope),
        )

    def forward(self, batch: Tensor) -> Tensor:
        """Pass `batch` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        Returns
        -------
        Tensor
            The result of passing `batch` through the model.

        """
        return self.conv2(self.conv1(batch))


class ResBlock(Module):
    """Residual block.

    Parameters
    ----------
    in_chans : int
        The number of input channels.

    """

    def __init__(self, in_chans: int):
        """Build `ResidualBlock`."""
        super().__init__()
        self.conv1 = ConvBlock(
            in_chans,
            in_chans,
            batch_norm=True,
            leaky_relu=True,
            lr_slope=0.0,
        )
        self.conv2 = ConvBlock(
            in_chans,
            in_chans,
            batch_norm=True,
            leaky_relu=False,
        )

        self.relu = ReLU()

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
        out = self.conv1(batch)
        out = self.conv2(out)
        out += identity
        return self.relu(out)


class UNetUpBlock(Module):
    """Upsampling block to be used in the second half of a UNet.

    Parameters
    ----------
    in_chans : int
        The number of input channels.
    out_chans : int
        The number of output channels.
    bilinear : bool
        If `True`, the upsample is done using bilinear interpolation using
        `torch.nn.Upsample`. Otherwise we use a `ConvTranspose2d`
    lr_slope : float
        The negative slope to use in the `LeakyReLU`.

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        bilinear: bool,
        lr_slope: float,
    ):
        """Build `UNetUpBlock`."""
        super().__init__()
        self._in_chans = self._process_in_chans(in_chans)
        self._out_chans = process_num_feats(out_chans)

        self._upsample = self._get_upsample(process_boolean_arg(bilinear))
        self._double_conv = DoubleConvBlock(
            self._in_chans,
            self._out_chans,
            lr_slope=lr_slope,
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

    def _get_upsample(self, bilinear: bool) -> Module:
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

        upsampled = self._upsample(to_upsample)

        height_diff = down_features.shape[2] - upsampled.shape[2]
        width_diff = down_features.shape[3] - upsampled.shape[3]

        padding = (
            width_diff // 2,  # Left padding
            width_diff - width_diff // 2,  # Right padding
            height_diff // 2,  # Top padding
            height_diff - height_diff // 2,  # Bottom padding
        )

        upsampled = pad(upsampled, padding)

        # Concatenate along the channel dimension (N, C, H, W)
        concatenated = cat([down_features, upsampled], dim=1)
        return self._double_conv(concatenated)
