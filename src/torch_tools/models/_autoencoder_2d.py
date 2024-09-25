"""A simple image encoder-decoder model."""

from torch.nn import Module

from torch import Tensor, set_grad_enabled

from torch_tools.models._encoder_2d import Encoder2d
from torch_tools.models._decoder_2d import Decoder2d

from torch_tools.models._argument_processing import (
    process_num_feats,
    process_u_architecture_layers,
    process_str_arg,
    process_negative_slope_arg,
    process_boolean_arg,
    process_2d_kernel_size,
    process_2d_block_style_arg,
    process_dropout_prob,
)

# pylint:disable=too-many-arguments, too-many-positional-arguments


class AutoEncoder2d(Module):
    """A simple encoder-decoder pair for image-like inputs.

    Parameters
    ----------
    in_chans : int
        The number of input channels.
    out_chans : int
        The number of output layers the model should produce.
    num_layers : int, optional
        The number of layers in the encoder/decoder.
    features_start : int, optional
        The number of features produced by the first conv block.
    lr_slope : float, optional
        The negative slope to use in the ``LeakyReLU`` layers.
    pool_style : str, optional
        The pool style to use in the downsampling blocks
        ( ``"avg"`` or ``"max"`` ).
    bilinear : bool, optional
        Whether or not to upsample with bilinear interpolation ( ``True`` ) or
        ``ConvTranspose2d`` ( ``False`` ).
    kernel_size : int, optional
        Size of the square convolutional kernel to use on the ``Conv2d``
        layers. Must be a positive, odd, int.
    block_style : str, optional
        Style of convolutional blocks to use in the encoding and decoding
        blocks.  Use either ``"double_conv"`` or ``"conv_res"``.
    dropout : float, optional
        The dropout probability to apply at the output of the convolutional
        blocks.


    Notes
    -----
    — Depending on the application, it may be convenient to pretrain this model
    and then use it for transfer learning—hence the ``frozen_encoder`` and
    ``frozen_decoder`` arguments in the ``forward`` method. There are no
    pretrained weights available, however.


    Examples
    --------
    >>> from torch_tools import AutoEncoder2d
    >>> model = AutoEncoder2d(
                    in_chans=3,
                    start_features=64,
                    num_blocks=4,
                    pool_style="max",
                    lr_slope=0.123,
                )

    Another (potentially) useful feature (if you want to do transfer learning)
    if the ability to *freeze*—i.e. fix—the parameters of either the encoder
    or the decoder:

    >>> from torch import rand
    >>> from torch_tools import AutoEncoder2d
    >>> # Mini-batch of ten, three-channel images of 64 by 64 pixels
    >>> mini_batch = rand(10, 3, 64, 64)
    >>> model = AutoEncoder2d(in_chans=3, out_chans=3)
    >>> # With nothing frozen (default behaviour)
    >>> pred = model(mini_batch, frozen_encoder=False, frozen_decoder=False)
    >>> # With the encoder frozen:
    >>> pred = model(mini_batch, frozen_encoder=True, frozen_decoder=False)
    >>> # With both the encoder and decoder frozen:
    >>> pred = model(mini_batch, frozen_encoder=True, frozen_decoder=True)

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        num_layers: int = 4,
        features_start: int = 64,
        lr_slope: float = 0.1,
        pool_style: str = "max",
        bilinear: bool = False,
        kernel_size: int = 3,
        block_style: str = "double_conv",
        dropout: float = 0.25,
    ):
        """Build ``EncoderDecoder2d``."""
        super().__init__()

        self.encoder = Encoder2d(
            process_num_feats(in_chans),
            process_num_feats(features_start),
            process_u_architecture_layers(num_layers),
            process_str_arg(pool_style),
            process_negative_slope_arg(lr_slope),
            process_2d_kernel_size(kernel_size),
            block_style=process_2d_block_style_arg(block_style),
            dropout=process_dropout_prob(dropout),
        )

        self.decoder = Decoder2d(
            process_num_feats((2 ** (num_layers - 1)) * features_start),
            process_num_feats(out_chans),
            process_u_architecture_layers(num_layers),
            process_boolean_arg(bilinear),
            process_negative_slope_arg(lr_slope),
            process_2d_kernel_size(kernel_size),
            block_style=process_2d_block_style_arg(block_style),
            dropout=process_dropout_prob(dropout),
        )

    def forward(
        self,
        batch: Tensor,
        frozen_encoder: bool = False,
        frozen_decoder: bool = False,
    ) -> Tensor:
        """Pass ``batch`` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.
        frozen_encoder : bool, optional
            Boolean switch controlling whether the encoder's gradients are
            enabled or disabled (useful for transfer learning).
        frozen_decoder : bool, optional
            Boolean switch controlling whether the decoder's gradients are
            enabled or disabled (useful for transfer learning).

        Returns
        -------
        Tensor
            The result of passing ``batch`` through the model.

        """
        with set_grad_enabled(not frozen_encoder):
            encoded = self.encoder(batch)

        with set_grad_enabled(not frozen_decoder):
            decoded = self.decoder(encoded)

        return decoded
