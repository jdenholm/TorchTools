"""2D convolutional variational autoencoder."""

from typing import Tuple, Union, Optional

from torch import (  # pylint: disable=no-name-in-module
    Tensor,
    flatten,
    unflatten,
    randn_like,
    set_grad_enabled,
)

from torch.nn import Module, Sequential, Conv2d

from torch_tools.models._encoder_2d import Encoder2d
from torch_tools.models._decoder_2d import Decoder2d

# from torch_tools.models._decoder_2d import Decoder2d
from torch_tools.models._fc_net import FCNet

from torch_tools.models._argument_processing import (
    process_num_feats,
    process_u_architecture_layers,
    process_str_arg,
    process_negative_slope_arg,
    process_2d_kernel_size,
    process_input_dims,
    process_optional_feats_arg,
)
from torch_tools.models._blocks_2d import DoubleConvBlock


class VAE2d(Module):  # pylint: disable=too-many-instance-attributes
    """2D convolutional variational autoencoder.

    Parameters
    ----------
    in_chans : int
        The number of input channels the model should take.
    out_chans : int
        The number of output channels the model should produce.
    input_dims : Tuple[int, int]
        The ``(height, width)`` of the input images (only necessary if
        ``mean_var_nets == "linear"``).
    start_features : int, optional
        The number of features the first double conv block should produce.
    num_layers : int, optional
        The number of layers in the U-like architecture.
    down_pool : str, optional
        The type of pooling to use in the down-sampling layers: ``"avg"`` or
        ``"max"``.
    bilinear : bool, optional
        If ``True``, we use bilinear interpolation in the upsampling. If
        ``False``, we use ``ConvTranspose2d``.
    lr_slope : float, optional
        Negative slope to use in the leaky relu layers.
    kernel_size : int, optional
        Linear size of the square convolutional kernels to use.
    max_down_feats : int, optional
        Upper limit on the number of features that can be produced by the
        down-sampling blocks.
    min_up_feats : int, optional
        Minimum number of features the up-sampling blocks can produce.
    block_style : str
        Block style to use in the down and up blocks.
    mean_var_net : str
        The style of the networks for which learn the mean and variances:
        ``"linear"`` or ``"conv"``.

    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        in_chans: int,
        out_chans: int,
        input_dims: Optional[Tuple[int, int]] = None,
        start_features: int = 64,
        num_layers: int = 4,
        down_pool: str = "max",
        bilinear: bool = False,
        lr_slope: float = 0.1,
        kernel_size: int = 3,
        max_down_feats: Optional[int] = None,
        min_up_feats: Optional[int] = None,
        block_style: str = "double_conv",
        mean_var_nets: str = "linear",
    ):
        """Build ``VAE2d``."""
        super().__init__()
        self.encoder = Encoder2d(
            in_chans=process_num_feats(in_chans),
            start_features=process_num_feats(start_features),
            num_blocks=process_u_architecture_layers(num_layers),
            pool_style=process_str_arg(down_pool),
            lr_slope=process_negative_slope_arg(lr_slope),
            kernel_size=process_2d_kernel_size(kernel_size),
            max_feats=process_optional_feats_arg(max_down_feats),
            block_style=block_style,
        )

        self._latent_chans, self._latent_feats = _latent_sizes(
            start_features,
            num_layers,
            process_input_dims(input_dims),
            max_down_feats,
        )

        self._input_dim_mean_var_net_check(input_dims, mean_var_nets)

        self._mean_var_style = mean_var_nets

        self._mean_var_funcs = {
            "linear": self._mean_logvar_linear,
            "conv": self._mean_logvar_conv,
        }

        self.mean_net = self._mean_or_var_net(lr_slope, kernel_size)
        self.var_net = self._mean_or_var_net(lr_slope, kernel_size)

        self.decoder = Decoder2d(
            in_chans=self._latent_chans,
            out_chans=process_num_feats(out_chans),
            num_blocks=num_layers,
            bilinear=bilinear,
            lr_slope=lr_slope,
            kernel_size=kernel_size,
            min_up_feats=min_up_feats,
            block_style=block_style,
        )

    def _input_dim_mean_var_net_check(
        self,
        input_dims: Union[Tuple[int, int], None],
        mean_var_nets: str,
    ):
        """Check ``input_dims`` and ``mean_var_nets`` compatibility.

        Parameters
        ----------
        input_dims : Tuple[int, int] or None
            The size of the input image.
        mean_var_nets : str
            The style of the mean/variance nets.

        """
        msg = "``input_dims`` should be ``None`` if ``mean_var_nets`` is "
        msg += "``''conv'`` and ``Tuple[int, int]`` if ``mean_var_nets`` is "
        msg += f"``'linear'``. Got '{mean_var_nets}' and '{input_dims}'."

        if mean_var_nets == "linear" and (not isinstance(input_dims, tuple)):
            raise ValueError(msg)
        if mean_var_nets == "conv" and (not isinstance(input_dims, type(None))):
            raise ValueError(msg)

    def _mean_or_var_net(
        self,
        lr_slope: float,
        kernel_size: int,
    ) -> Module:
        """Return a model for calculating the mean or variance.

        Parameters
        ----------
        lr_slope : float
            The negative slope aregument in the leaky relus.
        kernel_size : int
            The size of the kernel in the convolutional layers,

        Returns
        -------
        Module
            A network for learning the mean or standard deviation.

        Raises
        ------
        ValueError
            If ``mean_var_style`` is not ``"conv"`` or ``"linear"``.

        """
        if self._mean_var_style == "conv":
            return Sequential(
                DoubleConvBlock(
                    in_chans=process_num_feats(self._latent_chans),
                    out_chans=process_num_feats(self._latent_chans),
                    lr_slope=process_negative_slope_arg(lr_slope),
                    kernel_size=process_2d_kernel_size(kernel_size),
                ),
                Conv2d(
                    in_channels=process_num_feats(self._latent_chans),
                    out_channels=process_num_feats(self._latent_chans),
                    kernel_size=1,
                    stride=1,
                ),
            )
        if self._mean_var_style == "linear":
            return FCNet(
                in_feats=process_num_feats(self._latent_feats),  # type: ignore
                out_feats=process_num_feats(self._latent_feats),  # type: ignore
            )

        msg = f"mean_var_style '{self._mean_var_style}' not recognised. Choose"
        msg += " from 'conv' or 'linear'."
        raise ValueError(msg)

    def _mean_logvar_conv(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Estimate the mean and logvar vector.

        Parameters
        ----------
        features : Tensor
            Raw features from the encoders.

        Returns
        -------
        mean : Tensor
            The means.
        logvar : Tensor
            The logarithm of the variance.
        """
        mean = self.mean_net(features)
        logvar = self.var_net(features)

        return mean, logvar

    def _mean_logvar_linear(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Estimate the mean and logvar vector.

        Parameters
        ----------
        features : Tensor
            Raw features from the encoders.

        Returns
        -------
        mean : Tensor
            The means.
        logvar : Tensor
            The logarithm of the variance.

        """
        restore_shape = features.shape[1:]
        features = flatten(features, start_dim=1)
        mean, logvar = self.mean_net(features), self.var_net(features)

        # Restore shapes
        features = unflatten(features, dim=1, sizes=restore_shape)
        mean = unflatten(mean, dim=1, sizes=restore_shape)
        logvar = unflatten(logvar, dim=1, sizes=restore_shape)

        return mean, logvar

    def get_features(self, means: Tensor, logvar: Tensor) -> Tensor:
        """Get the features using the reparam trick.

        Parameters
        ----------
        means : Tensor
            The feature means.
        logvar : Tensor
            The log variance.

        Returns
        -------
        Tensor
            The feature dist.

        """
        return means + (randn_like(means) * (0.5 * logvar).exp())

    @staticmethod
    def kl_divergence(means: Tensor, log_var: Tensor) -> Tensor:
        """Compute the KL divergence between the dists and a unit normal.

        Parameters
        ----------
        means : Tensor
            Samples from the mean distributions.
        log_var : Tensor
            The logarithm of the variances.

        Returns
        -------
        Tensor
            Kullback-Leibler divergence between the feature dists and unit
            normals.

        """
        return -0.5 * (-log_var.exp() - means**2.0 + 1.0 + (log_var)).mean()

    def encode(
        self,
        batch: Tensor,
        frozen_encoder: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Encode the inputs in ``batch``.

        Parameters
        ----------
        batch : Tensor
            Mini-batch of inputs.
        frozen_encoder : bool
            Shoould the encoder's weights be frozen, or not?

        Returns
        -------
        feats : Tensor
            The encoded features.
        Tensor
            The KL divergence between the features and N(0, 1).

        """
        with set_grad_enabled(not frozen_encoder):
            encoder_feats = self.encoder(batch)

            mean, log_var = self._mean_var_funcs[self._mean_var_style](encoder_feats)

            feats = self.get_features(mean, log_var)

        return feats, self.kl_divergence(mean, log_var)

    def decode(
        self,
        features: Tensor,
        frozen_decoder: bool,
    ) -> Tensor:
        """Decode the latent ``features``.

        Parameters
        ----------
        features : Tensor
            VA-encoded features.
        frozen_decoder : bool
            Should the decoder's weights be frozen, or not?

        Returns
        -------
        Tensor
            The decoded ``features``.

        """
        with set_grad_enabled(not frozen_decoder):
            return self.decoder(features)

    def forward(
        self,
        batch: Tensor,
        frozen_encoder: bool = False,
        frozen_decoder: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Pass ``batch`` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of image-like inputs.
        frozen_encoder : bool, optional
            Should the encoder's parameters be fixed?
        frozen_decoder : bool, optional
            Should the decoder's weights be fixed?

        Returns
        -------
        decoded : Tensor
            The predicted version of ``batch``.
        kl_div : Tensor
            The KL divergence between ``features`` and N(0, 1).

        """
        features, kl_div = self.encode(batch, frozen_encoder)

        decoded = self.decode(features, frozen_decoder)

        return decoded, kl_div


def _latent_sizes(
    start_features: int,
    num_blocks: int,
    input_dims: Union[Tuple[int, int], None],
    max_feats: Optional[int] = None,
) -> Tuple[int, int]:
    """Get the size of the features produced by the encoder.

    Parameters
    ----------
    start_features : int
        The number features produced by the first block in the encoder.
    num_blocks : int
        The number of blocks in one half of the U-like architecture.
    input_dims : Tuple[int, int] or None
        The spatial dimensions of the model's inputs.
    max_feats : int, optional
        The maximum number of features allowed.

    Returns
    -------
    latent_chans : int
        The number of channels the image-like representation has after it is
        encoded.
    latent_feats : int
        The total number of features in the latent space after encoding. If
        the mean and var nets are linear, this is the number of channels.

    Raises
    ------
    ValueError
        If the number of features would be reduced to zero because
        ``input_dims`` is too small for the number of layers.


    """
    latent_chans = start_features
    for _ in range(num_blocks - 1):
        if max_feats is None:
            latent_chans *= 2
        else:
            latent_chans = min(max_feats, latent_chans * 2)

    if input_dims is None:
        latent_feats = latent_chans
    else:
        in_height, in_width = input_dims
        factor = 2 ** (num_blocks - 1)

        out_height = in_height / factor
        out_width = in_width / factor

        if (out_height % 1 != 0) or (out_width % 1 != 0):
            msg = f"Image dims '{(in_height, in_width)}' can't be halved {num_blocks - 1} times."
            raise ValueError(msg)

        latent_feats = int(out_height) * int(out_width) * latent_chans
        if latent_feats == 0:
            raise ValueError(f"{input_dims} too small for number of layers.")

    return latent_chans, latent_feats
