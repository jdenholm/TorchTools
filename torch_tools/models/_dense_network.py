"""A fully connected neural network model."""
from typing import Optional, Tuple, Union, List

from torch import Tensor
from torch.nn import Module, Sequential

from torch_tools.models._blocks_1d import DenseBlock, InputBlock

# pylint: disable=too-many-arguments


class DenseNetwork(Module):
    """Dense, fully connected neural network.

    Parameters
    ----------
    in_feats : int
        Number of input features to the model.
    out_feats : int
        Number of output features (classes).
    input_bnorm : bool, optional
        Should we apply batch-normalisation to the input batches?
    input_dropout : float, optional
        The dropout probability to apply to the inputs (not included if zero).
    hidden_sizes : Tuple[int], optional
        The sizes of the hidden layers (or None).
    hidden_dropout : float, optional
        The Dropout probability at each hidden layer (not included if zero).
    hidden_bnorm : bool, optional
        Should we include batch norms in the hidden layers?

    Examples
    --------
    TODO: add some examples.


    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        input_bnorm: bool = True,
        input_dropout: float = 0.25,
        hidden_sizes: Optional[Tuple[int, ...]] = None,
        hidden_dropout: float = 0.25,
        hidden_bnorm: bool = True,
    ):
        """Build `DenseClassifier`."""
        super().__init__()

        self._input_block = InputBlock(in_feats, input_bnorm, input_dropout)

        self._dense_blocks = self._list_dense_blocks(
            in_feats,
            out_feats,
            hidden_dropout,
            hidden_bnorm,
            hidden_sizes=hidden_sizes,
        )

    @staticmethod
    def _get_feature_sizes(
        in_feats: int,
        out_feats: int,
        hidden_sizes: Union[Tuple[int, ...], None],
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """List the input and output sizes of each block.

        Parameters
        ----------
        in_feats : int
            Number of input features to the network.
        out_feats : int
            Number of output features the model should produce.
        hidden_sizes : Tuple[int, ...] or None
            The sizes of the hidden layers.

        Returns
        -------
        Tuple[int, ...]
            Tuple of input feature sizes to each block.
        Tuple[int, ...]
            Tuple of output sizes for each block.

        """
        hidden_feats = hidden_sizes if hidden_sizes is not None else ()
        feature_sizes = (in_feats,) + hidden_feats + (out_feats,)
        return feature_sizes[:-1], feature_sizes[1:]

    def _list_dense_blocks(
        self,
        in_feats: int,
        out_feats: int,
        dropout_prob: float,
        batch_norms: bool,
        hidden_sizes: Optional[Tuple[int, ...]] = None,
    ) -> Sequential:
        """List the dense layers in the model.

        Parameters
        ----------
        in_feats : int
            Number of inputs features to this first DenseBlock.
        out_feats : int
            Number of output classes the final block should produce.
        dropout_prob : float
            The dropout probability (if zero, not included).
        batch_norms : bool
            Should we include batchnorms in the DenseBlocks?
        hidden_sizes : Tuple[int], optional
            Sizes of the hidden layers in the model.

        Returns
        -------
        Sequential
            All of the model's layers stacked in a `Sequential`.

        """
        in_sizes, out_sizes = self._get_feature_sizes(
            in_feats,
            out_feats,
            hidden_sizes,
        )
        finals = (len(in_sizes) - 1) * [False] + [True]

        blocks: List[Module]
        blocks = []
        for in_size, out_size, final in zip(in_sizes, out_sizes, finals):
            blocks.append(
                DenseBlock(
                    in_size,
                    out_size,
                    final_block=final,
                    dropout_prob=dropout_prob,
                    batch_norm=batch_norms,
                )
            )
        return Sequential(*blocks)

    def forward(self, batch: Tensor) -> Tensor:
        """Pass `batch` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        Returns
        -------
        Tensor
            The result of passing `batch` through the model.

        """
        input_layer_out = self._input_block(batch)
        return self._dense_blocks(input_layer_out)
