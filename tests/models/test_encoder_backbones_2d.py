"""Tests for `torch_tools.models._encoder_backbones`."""
import pytest


from torch_tools.models._encoder_backbones_2d import get_backbone


def test_get_backbone_with_bad_encoder_option_type():
    """Test `option` argument with bad type."""
    # Should work with str
    _ = get_backbone(option="resnet18")

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = get_backbone(option=123)
    with pytest.raises(TypeError):
        _ = get_backbone(option=["vgg11"])


def test_get_backbone_with_allowed_encoder_options():
    """Test allowed `option` arguments in `get_backbone`."""
    # Should work with resnet18, resnet34 and resnet50
    _ = get_backbone(option="resnet18")
    _ = get_backbone(option="resnet34")
    _ = get_backbone(option="resnet50")

    # Should work with vgg11, vgg13 and vgg16
    _ = get_backbone(option="vgg11")
    _ = get_backbone(option="vgg13")
    _ = get_backbone(option="vgg16")

    # Should work with vgg11_bn, vgg13_bn and vgg16_bn
    _ = get_backbone(option="vgg11_bn")
    _ = get_backbone(option="vgg13_bn")
    _ = get_backbone(option="vgg16_bn")


def test_get_backbone_with_non_allowed_options():
    """Test not-allowed `option` arguments in `get_backbone`."""
    with pytest.raises(RuntimeError):
        _ = get_backbone(option="Gandalf the Grey")
    with pytest.raises(RuntimeError):
        _ = get_backbone(option="Sauron the Deceiver")
