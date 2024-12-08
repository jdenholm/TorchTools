"""Main dataset object for `torch_tools`."""

from typing import Sequence, Union, Optional, Tuple
from pathlib import Path


from torch import Tensor, concat  # pylint: disable=no-name-in-module
from torchvision.transforms import Compose  # type: ignore


from numpy import ndarray
from numpy.random import default_rng


from torch_tools.datasets._base_dataset import _BaseDataset


# pylint: disable=too-many-arguments, too-few-public-methods
# pylint: disable=too-many-positional-arguments


class DataSet(_BaseDataset):
    """Completely custom and highly flexible dataset.

    Parameters
    ----------
    inputs : Sequence[str, Path, Tensor, ndarray]
        Inputs (or x items) for the dataset.
    targets : Optional[Sequence[str, Path, Tensor, ndarray]]
        Targets (or y items) for the dataset.
    input_tfms : Optional[Compose]
        A composition of transforms to apply to the inputs as they are
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
    mixup : bool
        Should we apply mixup augmentation? See the paper:
        https://arxiv.org/abs/1710.09412. If ``True``, we apply mixup, and the
        lambda parameter is sampled from a beta distribution with the
        parameters alpha=beta=0.4.

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
        mixup: bool = False,
    ):
        """Build `DataSet`."""
        super().__init__(inputs=inputs, targets=targets)
        self._x_tfms = self._receive_tfms(input_tfms)
        self._y_tfms = self._receive_tfms(target_tfms)
        self._both_tfms = self._receive_tfms(both_tfms)

        self._mixup = self._process_mixup_arg(mixup)

        if self._mixup is True:
            self._rng = default_rng(seed=123)

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

    @staticmethod
    def _process_mixup_arg(mixup: bool):
        """Process the ``mixup`` argument.

        Parameters
        ----------
        mixup : bool
            The boolean mixup switch.

        Returns
        -------
        mixup : bool
            The boolean mixup switch.

        Raises
        ------
        TypeError
            If ``mixup`` is not a bool.


        """
        if not isinstance(mixup, bool):
            raise TypeError(f"'mixup' should be boolean, got '{type(mixup)}'.")
        return mixup

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
        # type: ignore
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
        # type: ignore
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

    def _prepare_one_item(self, idx: int) -> Union[Tuple[Tensor, ...], Tensor]:
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

    def _apply_mixup(
        self,
        x_item: Tensor,
        y_item: Optional[Tensor] = None,
    ) -> Union[Tuple[Tensor, ...], Tensor]:
        """Apply the mixup augmentation.

        Parameters
        ----------
        x_item : Tensor
            The input item.
        y_item : Tensor, optional
            The target item.

        Returns
        -------
        x_item : Tensor
            An input item.
        y_item : Tensor, optional
            A target item.

        """
        if self._mixup is True:
            rand_idx = self._rng.integers(0, len(self))
            frac = self._rng.beta(0.4, 0.4)

        if y_item is None:
            if self._mixup is True:
                other_x = self._prepare_one_item(rand_idx)
                x_item = (frac * x_item) + ((1.0 - frac) * other_x)  # type: ignore

            return x_item

        if self._mixup is True:
            other_x, other_y = self._prepare_one_item(rand_idx)
            x_item = (frac * x_item) + ((1.0 - frac) * other_x)
            y_item = (frac * y_item) + ((1.0 - frac) * other_y)

        return x_item, y_item

    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, ...], Tensor]:
        """Return an input-target pair (or just an input).

        Parameters
        ----------
        idx : int
            Index of the item to return.

        """
        if self.targets is None:
            x_item = self._prepare_one_item(idx)
            x_item = self._apply_mixup(x_item)  # type: ignore
            return x_item

        x_item, y_item = self._prepare_one_item(idx)
        x_item, y_item = self._apply_mixup(x_item, y_item)

        return x_item, y_item
