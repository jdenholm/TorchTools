"""2D convolutional variational autoencoder."""
from typing import Tuple, Union, Optional

from torch import (  # pylint: disable=no-name-in-module
    Tensor,
    flatten,
    unflatten,
    randn_like,
    set_grad_enabled,
)

from torch.nn import Module

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


class VAE2d(Module):
    """2D convolutional variational autoencoder.

    Parameters
    ----------
    in_chans : int
        The number of input channels the model should take.
    out_chans : int
        The number of output channels the model should produce.
    input_dims : Tuple[int, int]
        The ``(height, width)`` of the input images.
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

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_chans: int,
        out_chans: int,
        input_dims: Tuple[int, int],
        start_features: int = 64,
        num_layers: int = 4,
        down_pool: str = "max",
        bilinear: bool = False,
        lr_slope: float = 0.1,
        kernel_size: int = 3,
        max_down_feats: Optional[int] = None,
        min_up_feats: Optional[int] = None,
    ):
        """Build ``VAE2d``."""
        super().__init__()
        _ = process_input_dims(input_dims)

        self.encoder = Encoder2d(
            in_chans=process_num_feats(in_chans),
            start_features=process_num_feats(start_features),
            num_blocks=process_u_architecture_layers(num_layers),
            pool_style=process_str_arg(down_pool),
            lr_slope=process_negative_slope_arg(lr_slope),
            kernel_size=process_2d_kernel_size(kernel_size),
            max_feats=process_optional_feats_arg(max_down_feats),
        )

        self._num_feats, self._num_chans = _features_size(
            start_features,
            num_layers,
            process_input_dims(input_dims),
            max_down_feats,
        )

        self._mean_net = FCNet(
            in_feats=self._num_feats,
            out_feats=self._num_feats,
        )

        self._var_net = FCNet(
            in_feats=self._num_feats,
            out_feats=self._num_feats,
        )

        self.decoder = Decoder2d(
            in_chans=self._num_chans,
            out_chans=process_num_feats(out_chans),
            num_blocks=num_layers,
            bilinear=bilinear,
            lr_slope=lr_slope,
            kernel_size=kernel_size,
            min_up_feats=min_up_feats,
        )

    def _get_mean_and_logvar(self, features: Tensor) -> Tuple[Tensor, Tensor]:
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
        flat_feats = flatten(features, start_dim=1)

        mean, logvar = self._mean_net(flat_feats), self._var_net(flat_feats)

        mean = unflatten(mean, dim=1, sizes=features.shape[1:])
        logvar = unflatten(logvar, dim=1, sizes=features.shape[1:])

        return mean, logvar

    def get_features(self, means: Tensor, logvar: Tensor, feats: Tensor) -> Tensor:
        """Get the features using the reparam trick.

        Parameters
        ----------
        means : Tensor
            The feature means.
        logvar : Tensor
            The log variance
        feats : Tensor
            The encoder features.

        Returns
        -------
        Tensor
            The feature dist.

        """
        return means + (randn_like(feats) * (0.5 * logvar).exp())

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
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Encode the inputs in ``batch``.

        Parameters
        ----------
        batch : Tensor
            Mini-batch of inputs.
        frozen_encoder : bool
            Shoould the encoder's weights be frozen, or not?

        """
        with set_grad_enabled(not frozen_encoder):
            encoder_feats = self.encoder(batch)

            means, log_var = self._get_mean_and_logvar(encoder_feats)

            feats = self.get_features(means, log_var, encoder_feats)

            if self.training is True:
                return feats, self.kl_divergence(means, log_var)

            return feats

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
        Union[Tensor, Tuple[Tensor, Tensor]]
            Either the decoded "image-like" object, if the model is in
            eval mode, or both the decoded image and the kl divergence
            between the latent vector and N(0, 1) if the model is in training
            mode.

        """
        if self.training is True:
            features, kl_div = self.encode(batch, frozen_encoder)
        else:
            features = self.encode(batch, frozen_encoder)

        decoded = self.decode(features, frozen_decoder)

        if self.training is True:
            return decoded, kl_div

        return decoded


def _features_size(
    start_features: int,
    num_blocks: int,
    input_dims,
    max_feats: Optional[int] = None,
) -> Tuple[int, int]:
    """Get the size of the features produced by the encoder.

    Parameters
    ----------
    start_features : int
        The number features produced by the first block in the encoder.
    num_blocks : int
        The number of blocks in one half of the U-like architecture.
    input_dims : int
        The spatial dimensions of the model's inputs.
    max_feats : int, optional
        The maximum number of features allowed.

    Returns
    -------
    features_size : int
        The total number of output features for the mean and std nets.
    out_chans : int
        The number of output channels.


    Raises
    ------
    ValueError
        If the number of features would be reduced to zero because
        ``input_dims`` is too small for the number of layers.


    """
    in_height, in_width = input_dims

    factor = 2 ** (num_blocks - 1)

    out_chans = start_features
    for _ in range(num_blocks - 1):
        if max_feats is None:
            out_chans *= 2
        else:
            out_chans = min(max_feats, out_chans * 2)

    out_height = in_height / factor
    out_width = in_width / factor

    if (out_height % 1 != 0) or (out_width % 1 != 0):
        msg = f"Image dims '{(in_height, in_width)}' can't be halved {num_blocks - 1} times."
        raise ValueError(msg)

    features_size = out_chans * int(out_height) * int(out_width)

    if features_size == 0:
        raise ValueError(f"{input_dims} too small for number of layers.")

    return features_size, out_chans
