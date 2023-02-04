"""Tests for ``torch_tools.weight_init``"""
import pytest

from torch.nn import Conv2d, Linear

from torchvision.models import resnet18

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


def test_applying_normal_init_to_resnet():
    """Should work when applied to a resnet."""
    model = resnet18(weights=None)
    model.apply(lambda x: weight_init.normal_init(x, mean=0.0, std=0.25))
