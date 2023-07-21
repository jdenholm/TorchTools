"""Test the iteration behaviours of `torch_tools.datasets.DataSet`."""
import pytest


from torch import zeros, ones, eye, rand  # pylint: disable=no-name-in-module
from torchvision.transforms import Compose  # type: ignore

from torch_tools.datasets import DataSet

# pylint: disable=redefined-outer-name


@pytest.fixture
def inputs_and_targets():
    """Create inputs and targets for testing with."""
    inputs = ["Fool", "of", "a", "Took"]
    targets = ["Throw", "yourself", "in", "next"]
    return inputs, targets


def test_inputs_only_iteration_returns(inputs_and_targets):
    """Test the correct values are returned with inputs only.

    Notes
    -----
    Test the correct input values are return with no targets or transforms.

    """
    inputs, _ = inputs_and_targets

    dataset = DataSet(inputs=inputs)

    for dset_return, input_item in zip(dataset, inputs):
        msg = "Incorrect return value in input-only iteration."
        assert dset_return == input_item, msg


def test_inputs_and_targets_return_values(inputs_and_targets):
    """Test the correct values with both inputs and targets."""
    inputs, targets = inputs_and_targets

    dataset = DataSet(inputs=inputs, targets=targets)

    for (dset_x, dset_y), ipt_item, tgt_item in zip(dataset, inputs, targets):
        msg = "Incorrect input item returned."
        assert dset_x == ipt_item, msg

        msg = "Incorrect target item returned"
        assert dset_y == tgt_item, msg


def test_input_transforms_are_applied(inputs_and_targets):
    """Test the input transforms are applied."""
    inputs, _ = inputs_and_targets

    input_tfms = Compose([lambda x: x + "modified"])

    dataset = DataSet(inputs=inputs, input_tfms=input_tfms)

    for dset_x, input_item in zip(dataset, inputs):
        msg = "Input transform not applied"
        assert dset_x == input_tfms(input_item), msg


def test_input_and_target_transforms_are_applied(inputs_and_targets):
    """Test the input and the target transforms are applied."""
    inputs, targets = inputs_and_targets

    input_tfms = Compose([lambda x: x + "input-modified"])
    target_tfms = Compose([lambda x: x + "target-modified"])

    dataset = DataSet(
        inputs=inputs,
        targets=targets,
        input_tfms=input_tfms,
        target_tfms=target_tfms,
    )

    for (x_item, y_item), in_item, tgt_item in zip(dataset, inputs, targets):
        msg = "Input transform not correctly applied."
        assert x_item == input_tfms(in_item), msg

        msg = "Target transforms not correctly applied."
        assert y_item == target_tfms(tgt_item), msg


def test_input_target_and_both_transforms_are_applied(inputs_and_targets):
    """Test the input, target and both transforms are applied in order."""
    inputs, targets = inputs_and_targets

    input_tfms = Compose([lambda x: zeros(3, 50, 50)])
    target_tfms = Compose([lambda x: ones(3, 50, 50)])
    both_tfms = Compose([lambda x: x + 10])

    dataset = DataSet(
        inputs=inputs,
        targets=targets,
        input_tfms=input_tfms,
        target_tfms=target_tfms,
        both_tfms=both_tfms,
    )

    for (x_item, y_item), in_item, tgt_item in zip(dataset, inputs, targets):
        msg = "Input then both transforms not applied"
        assert (x_item == both_tfms(input_tfms(in_item))).all(), msg

        msg = "Target then both transforms not applied."
        assert (y_item == both_tfms(target_tfms(tgt_item))).all(), msg


def test_len_method(inputs_and_targets):
    """Test the len method returns the correct values."""
    inputs, targets = inputs_and_targets

    msg = "Wrong length value."
    assert len(DataSet(inputs=inputs, targets=targets)) == len(inputs), msg


def test_iteration_with_no_mixup_and_inputs_only():
    """Test the dataset's iteration with no mixup."""
    inputs = list(ones(10, 2))

    dataset = DataSet(inputs=inputs)

    for x_item, input_item in zip(dataset, inputs):
        assert (x_item == input_item).all()


def test_iteration_with_no_mixup_and_inputs_and_targets():
    """Test the dataset's iteration with no mixup."""
    inputs = list(ones(10, 2))
    targets = [eye(2)[0] for _ in range(len(inputs))]

    dataset = DataSet(inputs=inputs, targets=targets)

    for (x_item, y_item), inpt_item, tgt_item in zip(dataset, inputs, targets):
        print(y_item, inpt_item)

        assert (x_item == inpt_item).all()
        assert (y_item == tgt_item).all()


def test_iteration_with_mixup_and_inputs_only():
    """Test the dataset's iteration with no mixup."""
    inputs = list(rand(100, 2))

    dataset = DataSet(inputs=inputs, mixup=True)

    equal = []
    for x_item, input_item in zip(dataset, inputs):
        equal.append((x_item == input_item).all())

    assert not all(equal)


def test_iteration_with_mixup_and_inputs_and_targets():
    """Test the dataset's iteration with no mixup."""
    inputs = list(rand(100, 2))
    targets = [rand(2) for _ in range(len(inputs))]

    dataset = DataSet(inputs=inputs, targets=targets, mixup=True)

    equal = []
    for (x_item, y_item), inpt_item, tgt_item in zip(dataset, inputs, targets):
        x_equal = (x_item == inpt_item).all()
        y_equal = (y_item == tgt_item).all()

        equal.append(x_equal and y_equal)

        print(y_item, tgt_item)
