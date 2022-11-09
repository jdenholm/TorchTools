"""Tests for the argument types of torch_tools.datasets.DataSet."""
from pathlib import Path
from string import ascii_lowercase

import numpy as np
import torch

import pytest

from torch_tools.datasets import DataSet


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
    _ = DataSet(inputs=list(torch.zeros(10, 2)))


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
    _ = DataSet(inputs=inputs, targets=list(torch.zeros(5, 2)))


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


def test_inputs_and_targets_with_mismatched_lengths():
    """Test `DataSet` catches mismatched input and target lengths."""
    # Should work when both are the same length
    inputs = ["Three", "rings", "for"]
    targets = ["the", "elven", "kings"]

    _ = DataSet(inputs=inputs, targets=targets)

    # Should break when the lengths don't match
    with pytest.raises(RuntimeError):
        _ = DataSet(inputs=inputs, targets=targets[:-1])
