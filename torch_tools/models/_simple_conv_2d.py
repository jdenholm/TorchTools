"""Simple two dimensional convolutional neural network."""

from torch.nn import Sequential, Linear

from torch_tools.models._encoder_2d import Encoder2d
from torch_tools.models._adaptive_pools_2d import get_adaptive_pool

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
        The style of downsampling pool to use in the encoder (`"avg"` or
        `"Max"`).
    adaptive_pool : str
        The style of adaptive pool to use on the encoder's output (`"avg"`,
        `"max"` or `"avg-max-concat"`.)
    lr_slope : float
        The negative slope to use in the `LeakReLU` layers.

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
    ):
        """Build `SimpleConvNet2d`."""
        super().__init__(
            Encoder2d(
                in_chans,
                features_start,
                num_blocks,
                downsample_pool,
                lr_slope,
            ),
            get_adaptive_pool(
                option=adaptive_pool,
                output_size=(1, 1),
            ),
            Linear(
                self._num_output_features(
                    num_blocks,
                    features_start,
                    adaptive_pool,
                ),
                out_feats,
            ),
        )

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
