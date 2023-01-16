"""Main dataset object for `torch_tools`."""
from typing import Sequence, Union, Optional, Tuple
from pathlib import Path


from torch import Tensor, concat  # pylint: disable=no-name-in-module
from torchvision.transforms import Compose


from numpy import ndarray


from torch_tools.datasets._base_dataset import _BaseDataset


# pylint: disable=too-many-arguments, too-few-public-methods


class DataSet(_BaseDataset):
    """Completely custom and highly flexible dataset.

    Parameters
    ----------
    inputs : Sequence[str, Path, Tensor, ndarray]
        Inputs (or x items) for the dataset.
    targets : Optional[Sequence[str, Path, Tensor, ndarray]]
        Targets (or y items) for the dataset.
    input_tfms : Optional[Compose]
        A composition of transfroms to apply to the inputs as they are
        selected.
    target_tfms : Optional[Compose]
        A composition of transforms to apply to the targets as they are
        selected.
    both_tfms : Optional[Compose]
        A composition of transforms to apply to both the input and target.
        Note: these transforms are applied after `input_tfms` and
        `target_tfms`, at which point the inputs and targets should be tensors.
        Each input--target pair will be concatenated along `dim=0`,
        transformed, and sliced apart, in the way one would apply rotations or
        reflections to images and segmentation masks. The dimensionality
        matters!

    Notes
    -----
    This dataset works for

        - Simple perceptron-style inputs.
        - Computer vision experiments, where the inputs are images.
        - Just about any problem where you need inputs, targets and custom
          transforms.
        - Doing inference (just set `targets=None` and it yields inputs only).

    """

    def __init__(
        self,
        inputs: Sequence[Union[str, Path, Tensor, ndarray]],
        targets: Optional[Sequence[Union[str, Path, Tensor, ndarray]]] = None,
        input_tfms: Optional[Compose] = None,
        target_tfms: Optional[Compose] = None,
        both_tfms: Optional[Compose] = None,
    ):
        """Build `DataSet`."""
        super().__init__(inputs=inputs, targets=targets)
        self._x_tfms = self._receive_tfms(input_tfms)
        self._y_tfms = self._receive_tfms(target_tfms)
        self._both_tfms = self._receive_tfms(both_tfms)

    @staticmethod
    def _receive_tfms(tfms: Optional[Compose] = None) -> Union[Compose, None]:
        """Check the transforms are `Compose` (or `None`) and return them.

        Parameters
        ----------
        tfms : Optional[Compose]
            The transfroms to check and return.

        Raises
        ------
        TypeError
            If `tfms` is not a `Compose`.

        """
        if not isinstance(tfms, (Compose, type(None))):
            msg = "Transforms should be wrapped in a 'Compose', or 'None'. "
            msg += f"Got '{type(tfms)}'."
            raise TypeError(msg)
        return tfms

    def _apply_input_tfms(
        self,
        x_item: Union[str, Path, Tensor, ndarray],
    ) -> Tensor:
        """Apply the input-only transforms.

        Parameters
        ----------
        x_item : Union[str, Path, Tensor, ndarray]
            The input item to be transformed.

        Returns
        -------
        Tensor
            `x_item` mapped to a tensor.

        """
        return self._x_tfms(x_item) if self._x_tfms is not None else x_item

    def _apply_target_transforms(
        self, y_item: Union[str, Path, Tensor, ndarray]
    ) -> Tensor:
        """Apply the target-only transforms.

        Parameters
        ----------
        y_item : Union[str, Path, Tensor, ndarray]
            A target item to transform.

        Returns
        -------
        Tensor
            `y_item` mapped to a tensor.

        """
        return self._y_tfms(y_item) if self._y_tfms is not None else y_item

    def _apply_both_tfms(
        self,
        x_item: Tensor,
        y_item: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply `self._both_tfms` to `x_item` and `y_item`.

        Parameters
        ----------
        x_item : Tensor
            Input item.
        y_item : Tensor
            Target item.

        Returns
        -------
        Tensor
            Input item.
        Tensor
            Target item.

        Notes
        -----
        `x_item` and `y_item` are concatenated along the channel dimension,
        transformed and then sliced apart.

        """
        if self._both_tfms is not None:
            slice_idx = x_item.shape[0]
            transformed = self._both_tfms(concat([x_item, y_item], dim=0))
            return transformed[:slice_idx], transformed[slice_idx:]
        return x_item, y_item

    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, ...], Tensor]:
        """Return an input-target pair (or just an input).

        Parameters
        ----------
        idx : int
            Index of the item to return.

        """
        x_item = self._apply_input_tfms(self.inputs[idx])

        if self.targets is None:
            return x_item

        y_item = self._apply_target_transforms(self.targets[idx])
        x_item, y_item = self._apply_both_tfms(x_item, y_item)

        return x_item, y_item
