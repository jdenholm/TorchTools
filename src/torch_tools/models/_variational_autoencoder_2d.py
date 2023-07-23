"""2D convolutional variational autoencoder."""
from typing import Tuple, Union

from torch import (  # pylint: disable=no-name-in-module
    Tensor,
    flatten,
    unflatten,
    randn_like,
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
)


class VAE2d(Module):
    """2D convolutional variational autoencoder."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_chans: int,
        input_dims: Tuple[int, int],
        start_features: int = 64,
        num_layers: int = 4,
        down_pool: str = "max",
        bilinear: bool = False,
        lr_slope: float = 0.1,
        kernel_size: int = 3,
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
        )

        self._num_feats = _features_size(
            start_features,
            num_layers,
            process_input_dims(input_dims),
        )

        self._mean_net = FCNet(
            in_feats=self._num_feats,
            out_feats=self._num_feats,
        )

        self._std_net = FCNet(
            in_feats=self._num_feats,
            out_feats=self._num_feats,
        )

        self._decoder = Decoder2d(
            in_chans=process_num_feats((2 ** (num_layers - 1)) * start_features),
            out_chans=in_chans,
            num_blocks=num_layers,
            bilinear=bilinear,
            lr_slope=lr_slope,
            kernel_size=kernel_size,
        )

    def _get_means_and_devs(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Estimate the means anmd standard deviations.

        Parameters
        ----------
        features : Tensor
            Raw features from the encoders.

        Returns
        -------
        mean : Tensor
            The means.
        std : Tensor
            The variances.

        """
        flat_feats = flatten(features, start_dim=1)

        mean, std = self._mean_net(flat_feats), self._std_net(flat_feats)

        mean = unflatten(mean, dim=1, sizes=features.shape[1:])
        std = unflatten(std, dim=1, sizes=features.shape[1:])

        return mean, std

    def get_features(self, means: Tensor, devs: Tensor, feats: Tensor) -> Tensor:
        """Get the features using the reparam trick.

        Parameters
        ----------
        means : Tensor
            The feature means.
        devs : Tensor
            The feature std devs.
        feats : Tensor
            The encoder features.

        Returns
        -------
        Tensor
            The feature dist.

        """
        return means + (randn_like(feats) * devs)

    def forward(self, batch: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Pass ``batch`` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of image-like inputs.


        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor]]
            Either the decoded "image-like" object, if the model is in
            eval mode, or both the decoded image and the kl divergence
            between the latent vector and N(0, 1) if the model is in training
            mode.

        """
        encoder_feats = self.encoder(batch)

        means, std = self._get_means_and_devs(encoder_feats)

        feats = self.get_features(means, std, encoder_feats)

        if self.training is True:
            return (
                self._decoder(feats),
                (std**2.0 + means**2.0 - std - 0.5).mean(),
            )

        return self._decoder(feats)


def _features_size(start_features: int, num_blocks: int, input_dims) -> int:
    """Get the size of the features produced by the encoder.

    Parameters
    ----------
    start_features : int
        The number features produced by the first block in the encoder.
    num_blocks : int
        The number of blocks in one half of the U-like architecture.
    input_dims : int
        The spatial dimensions of the model's inputs.

    Raises
    ------
    ValueError
        If the number of features would be reduced to zero because
        ``input_dims`` is too small for the number of layers.


    """
    in_height, in_width = input_dims

    factor = 2 ** (num_blocks - 1)

    out_feats = factor * start_features
    out_height = in_height // factor
    out_width = in_width // factor

    features_size = out_feats * out_height * out_width

    if features_size == 0:
        raise ValueError(f"{input_dims} too small for number of layers.")

    return features_size
