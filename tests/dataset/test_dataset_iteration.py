"""Test the iteration behaviours of `torch_tools.datasets.DataSet`."""
import pytest

from torch_tools.datasets import DataSet


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
