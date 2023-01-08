"""A simple image encoder-decoder model."""

from torch.nn import Module, Conv2d

from torch import Tensor, set_grad_enabled

from torch_tools import Encoder2d, Decoder2d
from torch_tools.models._blocks_2d import DoubleConvBlock

from torch_tools.models._argument_processing import process_num_feats


class EncoderDecoder2d(Module):
    """A simple encoder-decoder pair for image-like inputs.

    Parameters
    ----------
    Bla

    """

    def __init__(
        self,
        in_chans: int,
        num_layers: int = 4,
        features_start: int = 64,
        lr_slope: float = 0.1,
        pool_style: str = "max",
        bilinear: bool = False,
    ):
        """Build `EncoderDecoder2d`."""
        super().__init__()

        self._in_conv = DoubleConvBlock(
            process_num_feats(in_chans),
            process_num_feats(features_start),
            lr_slope,
        )

        self._encoder = Encoder2d(
            features_start,
            num_layers - 1,
            pool_style,
            lr_slope,
        )

        self._decoder = Decoder2d(
            (2 ** (num_layers - 1)) * features_start,
            num_layers - 1,
            bilinear,
            lr_slope,
        )

        self._out_conv = Conv2d(
            in_channels=features_start,
            out_channels=in_chans,  # Return should have `in_chans` channels.
            kernel_size=1,
            stride=1,
        )

    def forward(
        self,
        batch: Tensor,
        frozen_encoder: bool = False,
        frozen_decoder: bool = False,
    ) -> Tensor:
        """Pass `batch` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.
        frozen_encoder : bool
            Boolean switch controlling whether the encoder's gradients are
            enabled or disabled (useful for transfer learning).
        frozen_decoder : bool
            Boolean switch controlling whether the decoder's gradients are
            enabled or disabled (usefule for transfer learning).

        Returns
        -------
        Tensor
            The result of passing `batch` through the model.

        """
        with set_grad_enabled(not frozen_encoder):
            encoded = self._in_conv(batch)
            encoded = self._encoder(encoded)
        with set_grad_enabled(not frozen_decoder):
            decoded = self._decoder(encoded)
            decoded = self._out_conv(decoded)
        return decoded
