"""A fully connected classifier model."""
from typing import Optional, Tuple, List, Union

from torch import Tensor
from torch.nn import Module, Sequential

from torch_tools.models.blocks_1d import DenseBlock

# pylint: disable=too-many-arguments


class DenseNetwork(Module):
    """Dense classification model.

    Parameters
    ----------
    in_feats : int
        Number of input features to the model.
    out_feats : int
        Number of output features (classes).
    hidden_sizes : Tuple[int], optional
        The sizes of the hidden layers (or None).
    dropout_prob : float, optional
        The Dropout probability at each layer (Not included if zero).
    batchnorm : bool, optional
        Should we include batch norm layers in the dense blocks?

    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        hidden_sizes: Optional[Tuple[int, ...]] = None,
        dropout_prob: float = 0.1,
        batchnorm: bool = True,
    ):
        """Build `DenseClassifier`."""
        super().__init__()
        self._fwd_seq = self._get_dense_blocks(
            in_feats,
            out_feats,
            dropout_prob,
            batchnorm,
            hidden_sizes=hidden_sizes,
        )

    @staticmethod
    def _get_dense_blocks(
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
            The dropout probability (if zero, no included).
        batch_norms : bool
            Should we include batchnorms in the DenseBlocks?
        hidden_sizes : Tuple[int], optional
            Sizes of the hidden layers in the model.

        Returns
        -------
        Sequential
            All of the model's layers stacked in a `Sequential`.

        """
        hidden_feats = hidden_sizes if hidden_sizes is not None else []
        feature_sizes = [in_feats] + list(hidden_feats) + [out_feats]

        in_sizes = feature_sizes[:-1]
        out_sizes = feature_sizes[1:]
        finals = len(hidden_feats) * [False] + [True]
        print(finals)

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
        return self._fwd_seq(batch)
