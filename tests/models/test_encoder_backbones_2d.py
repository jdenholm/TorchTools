"""Tests for `torch_tools.models._encoder_backbones`."""
import pytest


from torch_tools.models._torchvision_encoder_backbones_2d import get_backbone
from torch_tools.models._torchvision_encoder_backbones_2d import _encoder_options


def test_get_backbone_with_bad_encoder_option_type():
    """Test `option` argument with bad type."""
    # Should work with str
    _, _, _ = get_backbone(option="resnet18")

    # Should break with non-str
    with pytest.raises(TypeError):
        _, _, _ = get_backbone(option=123)
    with pytest.raises(TypeError):
        _, _, _ = get_backbone(option=["vgg11"])


def test_get_backbone_with_allowed_encoder_options():
    """Test allowed `option` arguments in `get_backbone`."""
    # Should work with options in _encoder_options
    for option, _ in _encoder_options.items():
        # Note: setting pretrained to False to avoid downloading the weights
        _, _, _ = get_backbone(option=option, pretrained=False)


def test_get_backbone_with_non_allowed_options():
    """Test not-allowed `option` arguments in `get_backbone`."""
    with pytest.raises(ValueError):
        _, _, _ = get_backbone(option="Gandalf the Grey")
    with pytest.raises(ValueError):
        _, _, _ = get_backbone(option="Sauron the Deceiver")


def test_pretrained_argument_types():
    """Test the `pretrained` argument only works with bools."""
    # Should work with bool
    _, _, _ = get_backbone(option="resnet18", pretrained=True)
    _, _, _ = get_backbone(option="resnet18", pretrained=False)

    # Should break with non-bool
    with pytest.raises(TypeError):
        _, _, _ = get_backbone(option="resnet18", pretrained="True")
    with pytest.raises(TypeError):
        _, _, _ = get_backbone(option="resnet18", pretrained=1)
