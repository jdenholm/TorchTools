""""2D convolutional variational autoencoder."""
from typing import Tuple

from torch import Tensor
from torch.nn import Module

from torch_tools.models._encoder_2d import Encoder2d

# from torch_tools.models._decoder_2d import Decoder2d
from torch_tools.models._fc_net import FCNet

from torch_tools.models._argument_processing import (
    process_num_feats,
    process_u_architecture_layers,
    process_str_arg,
    process_negative_slope_arg,
    process_2d_kernel_size,
)


class VAE2d(Module):
    """2D convolutional variational autoencoder."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_chans: int,
        start_features: int,
        num_blocks: int,
        down_pool: str,
        lr_slope: float,
        kernel_size: int,
        input_dims: Tuple[int, int],
    ):
        """Build ``VAE2d``."""
        super().__init__()
        self._encoder = Encoder2d(
            in_chans=process_num_feats(in_chans),
            start_features=process_num_feats(start_features),
            num_blocks=process_u_architecture_layers(num_blocks),
            pool_style=process_str_arg(down_pool),
            lr_slope=process_negative_slope_arg(lr_slope),
            kernel_size=process_2d_kernel_size(kernel_size),
        )

        self._num_feats = _features_size(
            start_features,
            num_blocks,
            input_dims,
        )

        self._mean_net = FCNet(
            in_feats=self._num_feats,
            out_feats=self._num_feats,
            hidden_sizes=(self._num_feats,),
        )

        self._std_net = FCNet(
            in_feats=self._num_feats,
            out_feats=self._num_feats,
            hidden_sizes=(self._num_feats,),
        )

    def forward(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Pass ``batch`` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of image-like inputs.

        """
        features = self._encoder(batch)


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

    """
    in_height, in_width = input_dims

    factor = 2 ** (num_blocks - 1)

    out_feats = factor * start_features
    out_height = in_height // factor
    out_width = in_width // factor

    return out_feats * out_height * out_width
