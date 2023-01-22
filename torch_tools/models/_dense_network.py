"""A fully connected neural network model."""
from typing import Optional, Tuple, Union, List

from torch.nn import Module, Sequential

from torch_tools.models._blocks_1d import DenseBlock, InputBlock

from torch_tools.models._argument_processing import process_hidden_sizes

# pylint: disable=too-many-arguments


class DenseNetwork(Sequential):
    """Dense, fully connected neural network.

    An optional input block, which applies batch normalisation and dropout
    to the inputs, followed by a series of fully-connected blocks consisting
    of ``Linear``, ``BatchNorm1d`` and ``LeakyReLU`` layers, followed by a
    final ``Linear`` output layer.

    Parameters
    ----------
    in_feats : int
        Number of input features to the model.
    out_feats : int
        Number of output features (classes).
    hidden_sizes : Tuple[int, ...], optional
        The sizes of the hidden layers (or ``None``).
    input_bnorm : bool, optional
        Should we apply batch-normalisation to the input batches?
    input_dropout : float, optional
        The dropout probability to apply to the inputs (not included if zero).
    hidden_dropout : float, optional
        The Dropout probability at each hidden layer (not included if zero).
    hidden_bnorm : bool, optional
        Should we include batch norms in the hidden layers?
    negative_slope : float, optional
        The negative slope argument to use in the ``LeakyReLU`` layers.

    Examples
    --------
    >>> from torch_tools import DenseNetwork
    >>> DenseNetwork(in_feats=256,
                     out_feats=2,
                     hidden_sizes=(128, 64, 32),
                     input_bnorm=True,
                     input_dropout=0.1,
                     hidden_dropout=0.25,
                     hidden_bnorm=True,
                     negative_slope=0.2)

    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        hidden_sizes: Optional[Tuple[int, ...]] = None,
        input_bnorm: bool = False,
        input_dropout: float = 0.0,
        hidden_dropout: float = 0.25,
        hidden_bnorm: bool = True,
        negative_slope: float = 0.1,
    ):
        """Build `DenseClassifier`."""
        super().__init__(
            *self._list_all_blocks(
                in_feats,
                out_feats,
                process_hidden_sizes(hidden_sizes),
                input_bnorm,
                input_dropout,
                hidden_dropout,
                hidden_bnorm,
                negative_slope,
            )
        )

    def _list_all_blocks(
        self,
        in_feats: int,
        out_feats: int,
        hidden_sizes: Union[Tuple[int, ...], None],
        input_bnorm: bool,
        input_dropout: float,
        hidden_dropout: float,
        hidden_bnorm: bool,
        negative_slope: float,
    ) -> Sequential:
        """Put all of the model's blocks in a `Sequential`.

        Parameters
        ----------
        in_feats : int
            The number of input features.
        out_feats : int
            The number of output features the model should produce.
        hidden_sizes : List[int] or None
            The sizes of the hidden layers in the model.
        input_bnorm : bool
            Should we apply a batchnorm to the input?
        input_dropout : float
            Dropout probability to apply to the input (not included if 0.0).
        hidden_dropout : float
            The dropout probability to apply at the hidden layers (not
            included if 0.0).
        hidden_bnorm : bool
            Whether or not to apply batchnorm in the hidden layers.
        negative_slope : float
            The negative slope to use in the `LeakyReLU`s.

        Returns
        -------
        Sequential
            The model's blocks arranged in a `Sequential`.

        """
        in_block_list = self._input_block_in_list(
            in_feats,
            input_bnorm,
            input_dropout,
        )

        dense_block_list = self._list_dense_blocks(
            in_feats,
            out_feats,
            hidden_dropout,
            hidden_bnorm,
            negative_slope,
            hidden_sizes,
        )

        return Sequential(*in_block_list, *dense_block_list)

    def _input_block_in_list(
        self,
        in_feats: int,
        input_bnorm: bool,
        input_dropout: float,
    ) -> List[Module]:
        """Put the input block in a list if it's needed.

        Parameters
        ----------
        in_feats : int
            The number of inputs features the model should take.
        input_bnorm : bool
            Bool determining whether or not batchnorm should be applied to
            the input.
        input_dropout : float
            The dropout probability to apply to the input (no included if 0.0).

        Returns
        -------
        List[Module]
            The input block in a list, if needed, otherwise the list is empty.

        """
        if (input_bnorm is False) and (input_dropout == 0.0):
            return []
        return [InputBlock(in_feats, input_bnorm, input_dropout)]

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
        negative_slope: float,
        hidden_sizes: Union[Tuple[int, ...], None],
    ) -> List[Module]:
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
        negative_slope : float
            The negative slope to use in the leaky relu layers.
        hidden_sizes : Tuple[int]
            Sizes of the hidden layers in the model.

        Returns
        -------
        blocks
            All of the model's dense blocks in a list.

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
                    negative_slope=negative_slope,
                )
            )
        return blocks
