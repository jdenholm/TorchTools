"""Utilities for getting 2D convolutional encoder backbones."""
from torchvision import models
from torch.nn import Module, Sequential


_encoder_options = {
    "vgg11": models.vgg11,
    "vgg13": models.vgg13,
    "vgg16": models.vgg16,
    "vgg11_bn": models.vgg11_bn,
    "vgg13_bn": models.vgg13_bn,
    "vgg16_bn": models.vgg16_bn,
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}


def get_backbone(option: str, pretrained: bool = True) -> Module:
    """Return an encoder backbone.

    Parameters
    ----------
    option : str
        String specifying the encoder architecture to use.
    pretrained : bool
        Should we use ImageNet-pretrained weights?

    Returns
    -------
    Module
        The encoder part of the architecture (without pool).

    """
    _check_encoder_option_is_a_string(option)
    option = option.lower()
    _check_encoder_option_is_allowed(option)

    weights = "DEFAULT" if pretrained is True else None

    if "vgg" in option:
        full_vgg = _encoder_options[option](weights=weights)
        encoder = Sequential(full_vgg.features)
    if "resnet" in option:
        full_resnet = _encoder_options[option](weights=weights)
        # The resnet encoder is everything bar the final two children,
        # which are the pool and classification layers.
        encoder = Sequential(*list(full_resnet.children()))[:-2]
    return encoder


def _check_encoder_option_is_a_string(option: str):
    """Check the encoder option is a `str`.

    Parameters
    ----------
    option : str
        String specifying the encoder.

    Raises
    ------
    TypeError
        If `option` is not a `str`.

    """
    if not isinstance(option, str):
        msg = f"Encoder option should be a string. Got '{type(option)}'."
        raise TypeError(msg)


def _check_encoder_option_is_allowed(option: str):
    """Check the encoder option is allowed.

    Parameters
    ----------
    option : str
        The string giving the encoder option.

    Raises
    ------
    RuntimeError
        If the encoder option is not in `_encoder_options`.

    """
    if option not in _encoder_options:
        msg = f"Encoder option '{option}' not supported. Please choose from "
        msg += f"'{_encoder_options.keys()}'."
        raise RuntimeError(msg)
