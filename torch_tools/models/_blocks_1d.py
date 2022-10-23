"""One dimensional neural network blocks."""
from typing import List

from torch import Tensor
from torch.nn import Module, Sequential, Linear, BatchNorm1d, Dropout
from torch.nn import LeakyReLU

from torch_tools.models._argument_processing import process_num_feats
from torch_tools.models._argument_processing import process_boolean_arg
from torch_tools.models._argument_processing import process_dropout_prob
from torch_tools.models._argument_processing import process_negative_slope_arg

# pylint: disable=too-many-arguments


class DenseBlock(Module):
    """Fully connected dense block.

    Linear -> BatchNorm (optional) -> Dropout (optional)
    -> LeakyReLU (optional)

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
    final_block : bool
        If `True`, the block only includes a linear layer.
    negative_slope : float
        The negative slope to use in the `LeakyReLU`
        (set zero for normal ReLU).

    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        batch_norm: bool = True,
        dropout_prob: float = 0.5,
        final_block: bool = False,
        negative_slope: float = 0.2,
    ):
        """Build `FCBlock`."""
        super().__init__()
        self._fwd_seq = self._get_layers(
            process_num_feats(in_feats),
            process_num_feats(out_feats),
            process_boolean_arg(batch_norm),
            process_dropout_prob(dropout_prob),
            process_boolean_arg(final_block),
            process_negative_slope_arg(negative_slope),
        )

    @staticmethod
    def _get_layers(
        in_feats: int,
        out_feats: int,
        batch_norm: bool,
        dropout_prob: float,
        final: bool,
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
            Dropout probability (if zero, we don't include dropout).
        final : bool
            Should the batchnorm, dropout and activation be included?
        negative_slope : float
            The negative slope to use in the `LeakyReLU`.

        Returns
        -------
        Sequential
            The block's layers in a `Sequential`.

        """
        layer_list: List[Module]
        layer_list = [Linear(in_feats, out_feats)]

        if batch_norm is True and final is False:
            layer_list.append(BatchNorm1d(out_feats))

        if dropout_prob != 0.0 and final is False:
            layer_list.append(Dropout(p=dropout_prob))

        if final is False:
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


class InputBlock(Module):
    """Block for modifying the inputs before they pass through `DenseNetwork`.

    Parameters
    ----------
    in_feats : int
        The number of input features to the block.
    batch_norm : bool
        Should we apply batch-normalisation to the inputs?
    dropout : float
        Should we apply dropout to the inputs?

    """

    def __init__(self, in_feats: int, batch_norm: bool, dropout: float):
        """Build `InputBlock`."""
        super().__init__()
        self._fwd_seq = self._get_layers(
            process_num_feats(in_feats),
            process_boolean_arg(batch_norm),
            process_dropout_prob(dropout),
        )

    @staticmethod
    def _get_layers(
        in_feats: int,
        batch_norm: bool,
        dropout: float,
    ) -> Sequential:
        """Get the block's layers.

        Parameters
        ----------
        in_feats : int
            The number of input features to the block.
        batch_norm : bool
            Should we apply batch-normalisation to the inputs?
        dropout : float
            Should we apply dropout to the inputs?

        """
        layers: List[Module]
        layers = []
        if batch_norm is True:
            layers.append(BatchNorm1d(in_feats))
        if dropout != 0:
            layers.append(Dropout(p=dropout))
        return Sequential(*layers)

    def forward(self, batch: Tensor) -> Tensor:
        """Pass `batch` through the block.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs

        Returns
        -------
        Tensor
            The result of passing `batch` through the block.

        """
        return self._fwd_seq(batch)
