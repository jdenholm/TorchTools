"""Tests for the argument types of torch_tools.datasets.DataSet."""
from pathlib import Path
from string import ascii_lowercase

import numpy as np
import torch

from torch_tools.datasets import DataSet


def test_allowed_input_types():
    """Test `DataSet` works with the allowed input types."""
    strings = list(map(lambda x: str(x) + ".png", ascii_lowercase))
    print(strings)
    # Should work with `typing.Sequence` of:
    # `str`
    _ = DataSet(inputs=strings)
    # `Path`
    _ = DataSet(inputs=list(map(Path, strings)))
    # `numpy.ndarray`
    _ = DataSet(inputs=list(np.zeros((10, 2))))
    # `torch.Tensor`
    _ = DataSet(inputs=list(torch.zeros(10, 2)))
