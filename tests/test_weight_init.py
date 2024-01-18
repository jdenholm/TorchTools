"""Tests for ``torch_tools.weight_init``"""
from itertools import product

import pytest

from torch.nn import Conv2d, Linear

from torch_tools import weight_init


def test_normal_init_model_type_check():
    """Make sure the ``model`` argument only accepts ``Module``s."""
    # Should work with classes inheriting from torch.nn.Module
    weight_init.normal_init(Linear(10, 2))
    weight_init.normal_init(Conv2d(1, 1, 1))

    # Should fail with non-torch.nn.Module
    with pytest.raises(TypeError):
        weight_init.normal_init(123)


def test_normal_init_mean_type_check():
    """Test ``mean`` only accepts floats."""
    # Should work with floats
    weight_init.normal_init(Linear(10, 2), mean=0.0)

    # Should break with non-floats
    with pytest.raises(TypeError):
        weight_init.normal_init(Linear(10, 2), mean=0)
    with pytest.raises(TypeError):
        weight_init.normal_init(Linear(10, 2), mean=0j)


def test_normal_init_std_type_check():
    """Test ``std`` only accepts floats."""
    # Should work with floats
    weight_init.normal_init(Linear(10, 2), std=0.0)

    # Should break with non-floats
    with pytest.raises(TypeError):
        weight_init.normal_init(Linear(10, 2), std=0)
    with pytest.raises(TypeError):
        weight_init.normal_init(Linear(10, 2), std=0j)


def test_normal_init_attr_name_type_check():
    """Test ``attr_name`` only accepts str."""
    # Should work with str
    weight_init.normal_init(Linear(10, 2), attr_name="weight")
    weight_init.normal_init(Linear(10, 2), attr_name="bias")

    # Should break with non-str
    with pytest.raises(TypeError):
        weight_init.normal_init(Linear(10, 2), attr_name=666)

    with pytest.raises(TypeError):
        weight_init.normal_init(Linear(10, 2), attr_name=["Theoden of Rohan."])


def test_normal_init_is_applied():
    """Test ``normal_init`` is applied."""
    model = Linear(10**4, 10**4)

    for mean, std, attr_name in zip([0.0, 1.0], [1.0, 2.0], ["weight"]):
        weight_init.normal_init(model, attr_name=attr_name, mean=mean, std=std)

        assert getattr(model, attr_name).detach().mean().item() == pytest.approx(
            mean,
            abs=0.01,
        )

        assert getattr(model, attr_name).detach().std(
            correction=0
        ).item() == pytest.approx(
            std,
            abs=0.01,
        )
