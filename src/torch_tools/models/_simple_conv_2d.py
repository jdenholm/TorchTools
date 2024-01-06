"""A simple two-dimensional convolutional neural network."""
from typing import Optional, Dict, Any

from torch import Tensor
from torch.nn import Sequential, Flatten

from torch_tools.models._encoder_2d import Encoder2d
from torch_tools.models._adaptive_pools_2d import get_adaptive_pool
from torch_tools.models._fc_net import FCNet
from torch_tools.models._conv_net_2d import _forbidden_args_in_dn_kwargs
from torch_tools.models._argument_processing import process_num_feats
from torch_tools.models._argument_processing import process_2d_kernel_size
from torch_tools.models._argument_processing import process_2d_block_style_arg

# pylint: disable=too-many-arguments


class SimpleConvNet2d(Sequential):
    """A very simple 2D CNN with an encoder, pool, and fully-connected layer.

    Parameters
    ----------
    in_chans : int
        The number of input channels.
    out_feats : int
        The number of output features the fully connected layer should produce.
    features_start : int
        The number of features the input convolutional block should produce.
    num_blocks : int
        The number of encoding blocks to use.
    downsample_pool : str
        The style of downsampling pool to use in the encoder (``"avg"`` or
        ``"max"``).
    adaptive_pool : str
        The style of adaptive pool to use on the encoder's output (``"avg"``,
        ``"max"`` or ``"avg-max-concat"``.)
    lr_slope : float
        The negative slope to use in the ``LeakyReLU`` layers.
    kernel_size : int
        The size of the square convolutional kernel to use in the ``Conv2d``
        layers. Must be an odd, positive, int.
    fc_net_kwargs : Dict[str, Any], optional
        Keyword arguments for ``torch_tools.models.fc_net.FCNet`` which serves
        as the classification/regression part of the model.
    block_style : str, optional
        Style of encoding blocks to use: choose from ``"double_conv"`` or
        ``conv_res``.

    Examples
    --------
    >>> from torch_tools import SimpleConvNet2d
    >>> SimpleConvNet2d(
            in_chans=3,
            out_feats=128,
            features_start=64,
            num_blocks=4,
            downsample_pool="max",
            adaptive_pool="avg-max-concat",
            lr_slope=0.123,
            fc_net_kwards={"hidden_sizes": (256, 256,)},
        )

    """

    def __init__(
        self,
        in_chans: int,
        out_feats: int,
        features_start: int = 64,
        num_blocks: int = 4,
        downsample_pool: str = "max",
        adaptive_pool: str = "avg",
        lr_slope: float = 0.1,
        kernel_size: int = 3,
        fc_net_kwargs: Optional[Dict[str, Any]] = None,
        block_style: str = "double_conv",
    ):
        """Build ``SimpleConvNet2d``."""
        encoder_feats = self._num_output_features(
            num_blocks,
            process_num_feats(features_start),
            adaptive_pool,
        )

        if fc_net_kwargs is not None:
            _forbidden_args_in_dn_kwargs(fc_net_kwargs)
            self._dn_args.update(fc_net_kwargs)

        super().__init__(
            Encoder2d(
                process_num_feats(in_chans),
                process_num_feats(features_start),
                num_blocks,
                downsample_pool,
                lr_slope,
                process_2d_kernel_size(kernel_size),
                block_style=process_2d_block_style_arg(block_style),
            ),
            get_adaptive_pool(
                option=adaptive_pool,
                output_size=(1, 1),
            ),
            Flatten(),
            FCNet(
                in_feats=encoder_feats,
                out_feats=out_feats,
                **self._dn_args,
            ),
        )

    def get_features(self, batch: Tensor) -> Tensor:
        """Get the features produced by the encoder and adaptive poool.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of image-like inputs.

        Returns
        -------
        Tensor
            The features for each item in ``batch``.

        """
        feats = self[0](batch)
        feats = self[1](feats)
        return self[2](feats)

    _dn_args: Dict[str, Any] = {
        "hidden_sizes": None,
        "input_bnorm": False,
        "hidden_bnorm": False,
        "input_dropout": 0.0,
        "hidden_dropout": 0.0,
        "negative_slope": 0.2,
    }

    def _num_output_features(
        self,
        num_blocks: int,
        features_start: int,
        adaptive_pool_style: str,
    ) -> int:
        """Get the number of output features the adaptive pool will produce.

        Parameters
        ----------
        num_blocks : int
            Number of layers argument from the user.
        features_start : int
            Number of features produced by the first convolutional block.
        adaptive_pool_style : str
            The user-requested adaptive pool style.

        Returns
        -------
        int
            The number of features the adaptive pooling layer will produce.

        """
        feats = (2 ** (num_blocks - 1)) * features_start
        return feats * 2 if adaptive_pool_style == "avg-max-concat" else feats
