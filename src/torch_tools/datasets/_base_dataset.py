"""Base dataset class."""
from pathlib import Path
from typing import Tuple, Union, Sequence, Optional

from collections import Counter

from numpy import ndarray

from torch import Tensor
from torch.utils.data import Dataset


class _BaseDataset(Dataset):
    """Base dataset class.

    Parameters
    ----------
    inputs : Sequence[Union[str, Path, Tensor, ndarray]]
        Inputs for a model. The inputs can be torch tensors, numpy arrays or
        paths to files which will be loaded and converted to tensors by
        downstream transforms.
    targets : Optional[Sequence[Union[str, Path, Tensor, ndarray]]] = None
        The targets (or ground truths) of the dataset. `targets` can be
        any of the allowed type options for `inputs`, or `None`. If `None`,
        `self.__getitem__` simply returns inputs (useful for inference).

    Notes
    -----
    Peforms type and length checks on the inputs and targets, to make sure
    they are acceptable and match in length.

    """

    def __init__(
        self,
        inputs: Sequence[Union[str, Path, Tensor, ndarray]],
        targets: Optional[Sequence[Union[str, Path, Tensor, ndarray]]] = None,
    ):
        """Build `_BaseDataset`."""
        self.inputs = self._set_inputs(inputs)
        self.targets = self._set_targets(targets)

        self._check_lengths()

    _allowed_types = (str, Path, Tensor, ndarray)

    def _check_lengths(self):
        """Check the lengths of images and targets match.

        Raises
        ------
        RuntimeError
            If the length of `self.inputs` and `self.targets` do not match
            (and `self.targets` is not None).

        """
        if self.targets is None:
            return
        if len(self.inputs) != len(self.targets):
            msg = "'inputs' and 'targets' should have same length. 'inputs' "
            msg += f"has length {len(self.inputs)} and 'targets' has length "
            msg += f"{len(self.targets)}."
            raise RuntimeError(msg)

    def _set_inputs(
        self,
        inputs: Sequence[Union[str, Path, Tensor, ndarray]],
    ) -> Tuple[Union[str, Path, Tensor, ndarray], ...]:
        """Set the dataset's inputs.

        Parameters
        ----------
        inputs : Sequence[Union[str, Path, Tensor, ndarray]]
            See class docstring.

        Returns
        -------
        Tuple[Union[str, Path, Tensor, ndarray]]
            The inputs in a tuple.

        """
        self._input_type(inputs)
        self._individual_types(inputs)
        return tuple(inputs)

    def _set_targets(
        self,
        targets: Optional[Sequence[Union[str, Path, Tensor, ndarray]]] = None,
    ) -> Union[Tuple[Union[str, Path, Tensor, ndarray], ...], None]:
        """Set the targets (ground truths) of the dataset.

        Parameters
        ----------
        targets : Optional[Sequence[Union[str, Path, Tensor, ndarray]]] = None
            See class docstring.

        Returns
        -------
        Union[Tuple[Union[str, Path, Tensor, ndarray], ...], None]
            `targets` in a tuple, or `None`.

        """
        if targets is None:
            return targets
        self._input_type(targets)
        self._individual_types(targets)

        return tuple(targets)

    @staticmethod
    def _input_type(inputs: Sequence[Union[str, Path, Tensor, ndarray]]):
        """Make sure `inputs` is a `Sequence`.

        Parameters
        ----------
        inputs : Sequence[Union[str, Path, Tensor, ndarray]]
            See class docstring.

        Raises
        ------
        TypeError
            If `inputs` is not a `Sequence`.

        """
        if not isinstance(inputs, Sequence):
            msg = f"Dataset inputs should be a Sequence. Got '{type(inputs)}'."
            raise TypeError(msg)

    def _individual_types(
        self,
        inputs: Sequence[Union[str, Path, Tensor, ndarray]],
    ):
        """Check the types of the individual items in `inputs`.

        Parameters
        ----------
        inputs : Sequence[Union[str, Path, Tensor, ndarray]]
            See class docstring.

        Raises
        ------
        TypeError
            If any of the inputs types are not in `self._allowed_types`, or
            there is more than one unique input type.

        """
        input_types = list(map(type, inputs))
        unique_types = list(Counter(input_types))
        all_allowed = all(map(lambda x: isinstance(x, self._allowed_types), inputs))

        if not (len(unique_types) == 1 and all_allowed):
            msg = "Expected one unique input type from "
            msg += f"'{self._allowed_types}'. Instead got types "
            msg += f"'{unique_types}'."
            raise TypeError(msg)

    def __len__(self):
        """Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, ...], Tensor]:
        """Return input or input--target pair.

        Parameters
        ----------
        idx : int
            Index of the item in the datset to retun.

        """
        msg = "__getitem__ method of dataset must be overloaded."
        raise NotImplementedError(msg)
