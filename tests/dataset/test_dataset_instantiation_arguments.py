"""Tests for the argument types of torch_tools.datasets.DataSet."""
from pathlib import Path
from string import ascii_lowercase

import numpy as np
from torch import zeros  # pylint: disable=no-name-in-module

import pytest

from torchvision.transforms import Compose  # type: ignore

from torch_tools.datasets import DataSet

# pylint: disable=redefined-outer-name


@pytest.fixture
def inputs_and_targets():
    """Return lists of strings for inputs and targets."""
    return ["Three", "rings", "for"], ["the", "elven", "kings"]


def test_inputs_with_allowed_types():
    """Test `DataSet` accepts `inputs` with allowed types."""
    strings = list(map(lambda x: str(x) + ".png", ascii_lowercase))
    # Should work with `typing.Sequence` of:
    # `str`
    _ = DataSet(inputs=strings)
    # `Path`
    _ = DataSet(inputs=list(map(Path, strings)))
    # `numpy.ndarray`
    _ = DataSet(inputs=list(np.zeros((10, 2))))
    # `torch.Tensor`
    _ = DataSet(inputs=list(zeros(10, 2)))


def test_inputs_with_forbidden_types():
    """Test `DataSet` catches `inputs` with forbidden types."""
    # Should break with non-sequence
    with pytest.raises(TypeError):
        _ = DataSet(inputs=1)
    with pytest.raises(TypeError):
        _ = DataSet({"Not sequence": 1})

    # Should break with Sequence containing which doesn't contain
    # 'str', 'Path', 'numpy.ndarray', 'torch.Tensor'
    with pytest.raises(TypeError):
        _ = DataSet(inputs=[1, 2, 3, 4, 5, 6])
    with pytest.raises(TypeError):
        _ = DataSet(inputs=(((1,)), (2,), (3,)))


def test_inputs_with_inconsistent_allowed_types():
    """Test `DataSet` catches when `inputs` has inconsisten types."""
    # Should break with allowed types if they are inconsistent
    with pytest.raises(TypeError):
        _ = DataSet(inputs=["mixed", Path("inputs")])


def test_targets_with_allowed_types():
    """Test `DataSet`'s `targets` argument accepts allowed types."""
    inputs = ["allowed", "inputs", "of", "type", "str"]
    # Should work with `typing.Sequence` of:
    # `str`
    _ = DataSet(inputs=inputs, targets=["1", "2", "3", "4", "5"])
    # `Path`
    _ = DataSet(
        inputs=inputs,
        targets=[Path("1"), Path("2"), Path("3"), Path("4"), Path("5")],
    )
    # `numpy.ndarray`
    _ = DataSet(inputs=inputs, targets=list((np.zeros((5, 3)))))
    # `torch.Tensor`
    _ = DataSet(inputs=inputs, targets=list(zeros(5, 2)))


def test_targets_with_forbidden_types():
    """Test `DataSet`'s `targets` argument catches forbidden types."""
    inputs = ["allowed", "types", "for", "testing"]

    # Should break with non-sequence
    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=1)
    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets={"not a": "sequence"})

    # Should break with a Sequence of the wrong type
    # I.e. not a str, path, numpy.ndarray or torch.Tensor.
    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=[1, 2, 3, 4])
    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=((1,), (2,), (3,), (4,)))


def test_target_args_with_inconsisten_allowed_types():
    """Test `DataSet` with allowed but inconsistent target types."""
    inputs = ["all", "allowed", "inputs"]

    # Should break with a Sequence of allowed, but inconsistent, types.
    with pytest.raises(TypeError):
        _ = DataSet(
            inputs=inputs,
            targets=["inconsistent", "but", Path("allowed")],
        )


def test_inputs_and_targets_with_mismatched_lengths(inputs_and_targets):
    """Test `DataSet` catches mismatched input and target lengths."""
    # Should work when both are the same length
    inputs, targets = inputs_and_targets

    _ = DataSet(inputs=inputs, targets=targets)

    # Should break when the lengths don't match
    with pytest.raises(RuntimeError):
        _ = DataSet(inputs=inputs, targets=targets[:-1])


def test_input_transforms_types(inputs_and_targets):
    """Test the types allowed and rejected by `input_tfms`."""
    inputs, targets = inputs_and_targets
    input_tfms = Compose([])

    # Should work with torchvision.transforms.Compose or `None`
    _ = DataSet(inputs=inputs, targets=targets, input_tfms=input_tfms)
    _ = DataSet(inputs=inputs, targets=targets, input_tfms=None)

    # Should break with any non-`Compose`
    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=targets, input_tfms=1)
    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=targets, input_tfms=lambda x: x)


def test_target_transforms_types(inputs_and_targets):
    """Test accepted and rejected types for `target_tfms` arg."""
    inputs, targets = inputs_and_targets

    # Should work with `torchvision.transforms.Compose` or `None`.
    _ = DataSet(inputs=inputs, targets=targets, target_tfms=Compose([]))
    _ = DataSet(inputs=inputs, targets=targets, target_tfms=None)

    # Should break with any non-`Compose`.
    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=targets, target_tfms=1)
    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=targets, target_tfms=lambda x: x)


def test_both_transforms_types(inputs_and_targets):
    """Test accepted and rejected types of `both_tfms` type."""
    inputs, targets = inputs_and_targets

    # Should work with `torchvision.transforms.Compose` or `None`.
    _ = DataSet(inputs=inputs, targets=targets, both_tfms=Compose([]))
    _ = DataSet(inputs=inputs, targets=targets, both_tfms=None)

    # Should break with any non-`Compose`.
    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=targets, both_tfms="Frodo")
    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=targets, both_tfms=lambda x: x)


def test_dataset_mixup_argument_types(inputs_and_targets):
    """Test the types accepted by the ``mixup`` arguments."""
    inputs, targets = inputs_and_targets

    # Should work with bools
    _ = DataSet(inputs=inputs, targets=targets, mixup=True)
    _ = DataSet(inputs=inputs, targets=targets, mixup=False)

    # Should break with non-bool
    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=targets, mixup=1)

    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=targets, mixup=1.0)

    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=targets, mixup=1j)

    with pytest.raises(TypeError):
        _ = DataSet(inputs=inputs, targets=targets, mixup="Grima Wormtongue")
