"""One dimensional neural network blocks."""
from typing import List

from torch import Tensor
from torch.nn import Module, Sequential, Linear, BatchNorm1d, Dropout
from torch.nn import LeakyReLU

# pylint: disable=too-many-arguments


class DenseBlock(Module):
    """Fully connected block.

    Parameters
    ----------
    in_feats : int
        Number of input features to the block.
    out_feats : int
        Number of output features to the block.
    batch_norm : bool
        Should we add a batch norm to the block.
    dropout_prob : float
        The dropout probability (won't be included if zero).
    neagtive_slope : float
        The negative slope to use in the `LeakyReLU`.

    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        batch_norm: bool = True,
        dropout_prob: float = 0.5,
        negative_slope: float = 0.2,
    ):
        """Build `FCBlock`."""
        super().__init__()
        self._fwd_seq = self._get_layers(
            in_feats, out_feats, batch_norm, dropout_prob, negative_slope,
        )

    @staticmethod
    def _get_layers(
        in_feats: int,
        out_feats: int,
        batch_norm: bool,
        dropout_prob: float,
        negative_slope: float,
    ) -> Sequential:
        """Return the block's layers in a `Sequential`.

        Parameters
        ----------
        in_feats : int
            Number of input features to the block.
        out_feats : int
            Number of output features the block should produce.
        batch_norm : bool
            Should we include a batchnorm after the linear layer?
        dropout_prob : float
            Dropout probability (if zero, we don't include the layer).
        negative_slope : float
            The negative slope to use in the `LeakyReLU`.

        Returns
        -------
        Sequential
            The block's layers in a `Sequential`.

        """
        layer_list: List[Module]
        layer_list = [Linear(in_feats, out_feats)]

        if batch_norm is True:
            layer_list.append(BatchNorm1d(out_feats))

        if dropout_prob != 0.0:
            layer_list.append(Dropout(p=dropout_prob))

        layer_list.append(LeakyReLU(negative_slope=negative_slope))

        return Sequential(*layer_list)

    def forward(self, batch: Tensor) -> Tensor:
        """Pass `batch` through the block.

        Parameters
        ----------
        batch : Tensor
            Mini-batch of inputs.

        Returns
        -------
        Tensor
            The result of passing `batch` through the block.

        """
        return self._fwd_seq(batch)
