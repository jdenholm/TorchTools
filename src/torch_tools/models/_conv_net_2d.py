"""2D CNN model which wraps Torchvision's ResNet and VGG models."""
from typing import Dict, Any, Optional

from torch import Tensor, set_grad_enabled
from torch.nn import Module, Sequential, Flatten, Conv2d

from torch_tools.models._argument_processing import process_num_feats
from torch_tools.models._torchvision_encoder_backbones_2d import get_backbone
from torch_tools.models._adaptive_pools_2d import get_adaptive_pool
from torch_tools.models._fc_net import FCNet

# pylint: disable=too-many-arguments


class ConvNet2d(Module):
    """CNN model which wraps Torchvision's ResNet, VGG and Mobilenet_v3 models.

    The model contains:
        — An encoder, taken from Torchvision's ResNet/VGG models.

        — An adaptive pooling layer.

        — A fully-connected classification/regression head.

    Parameters
    ----------
    out_feats : int
        The number of output features the model should produce (for example,
        the number of classes).
    in_channels : int
        Number of input channels the model should take. Warning: if you don't
        use three input channels, the first conv layer is overwritten, which
        renders freezing the encoder pointless.
    encoder_style : str, optional
        The encoder option to use. The encoders are loaded from torchvision's
        models. Options include all of torchvision's VGG, ResNET and MOBILENET
        v3 options (i.e. ``"vgg11"``, ``"vgg11_bn"``, ``"resnet18"``,
        ``mobilenet_v3_small`` etc.).
    pretrained : bool, optional
        Determines whether the encoder is initialised with Torchvision's
        pretrained weights. If ``True``, the model will load Torchvision's most
        up-to-date image-net-trained weights.
    pool_option : str, optional
        The type of adaptive pooling layer to use. Choose from ``"avg"``,
        ``"max"`` or ``"avg-max-concat"`` (the latter simply concatenates the
        former two). See ``torch_tools.models._adaptive_pools_2d`` for more
        info.
    fc_net_kwargs : Dict[str, Any], optional
        Keyword arguments for
        ``torch_tools.models.fc_net.FCNet`` which serves as the
        classification/regression part of the model.

    Examples
    --------
    >>> from torch_tools import ConvNet2d
    >>> model = ConvNet2d(out_feats=512,
                          in_channels=3,
                          encoder_style="vgg11_bn",
                          pretrained=True,
                          pool_style="avg-max-concat",
                          fc_net_kwargs={"hidden_sizes": (1024, 1024), "hidden_dropout": 0.25})


    Another potentially useful feature is the ability to *freeze* the encoder,
    and take advantage of the available pretrained weights by doing transfer
    learning.

    >>> from torch import rand
    >>> from torch_tools import ConvNet2d
    >>> model = ConvNet2d(out_feats=10, pretrained=True)
    >>> # Batch of 10 fake three-channel images of 256x256 pixels
    >>> mini_batch = rand(10, 3, 256, 256)
    >>> # With the encoder frozen
    >>> preds = model(mini_batch, frozen_encoder=True)
    >>> # Without the encoder frozen (default behaviour)
    >>> preds = model(mini_batch, frozen_encoder=False)


    Notes
    -----
    — Even if you load pretrained weights, but *don't* freeze the encoder, you
    will likely end up finding better performance than you would by randomly
    initialising the model—even if it doesn't make sense. Welcome to deep
    learning.

    — If you change the number of input channels, don't bother freezing the
    encoder—the first convolutional layer is overloaded and randomly
    initialised.

    — See ``torch_tools.models._conv_net_2d.ConvNet2d`` for more info.

    """

    def __init__(
        self,
        out_feats: int,
        in_channels: int = 3,
        encoder_style: str = "resnet34",
        pretrained=True,
        pool_style: str = "avg-max-concat",
        fc_net_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Build `ConvNet2d`."""
        super().__init__()
        self.backbone, num_feats, pool_size = get_backbone(
            encoder_style,
            pretrained=pretrained,
        )

        self._replace_first_conv_if_necessary(process_num_feats(in_channels))

        self.pool = Sequential(
            get_adaptive_pool(pool_style, pool_size),
            Flatten(),
        )

        if fc_net_kwargs is not None:
            _forbidden_args_in_dn_kwargs(fc_net_kwargs)
            self._dn_args.update(fc_net_kwargs)

        self.dense_layers = FCNet(
            2 * num_feats if pool_style == "avg-max-concat" else num_feats,
            process_num_feats(out_feats),
            **self._dn_args,
        )

    _dn_args: Dict[str, Any]
    _dn_args = {
        "hidden_sizes": None,
        "input_bnorm": False,
        "hidden_bnorm": False,
        "input_dropout": 0.0,
        "hidden_dropout": 0.0,
        "negative_slope": 0.2,
    }

    def _replace_first_conv_if_necessary(self, in_channels: int):
        """Replace the first conv layer if input channels don't match.

        Parameters
        ----------
        in_channels : int
            The number of input channels requested by the user.

        """
        for _, module in self.backbone.named_children():
            if isinstance(module, Conv2d):
                config = _conv_config(module)

                if config["in_channels"] != in_channels:
                    config["in_channels"] = in_channels
                    setattr(self.backbone, _, Conv2d(**config))  # type:ignore
                break

    def forward(self, batch: Tensor, frozen_encoder: bool = False) -> Tensor:
        """Pass `batch` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs with shape (N, C, H, W), where N is the
            batch-size, C the number of channels and (H, W) the input
            size.
        frozen_encoder : bool, optional
            If ``True``, the gradients are disabled in the encoder. If
            ``False``, the gradients are enabled in the encoder.

        Returns
        -------
        Tensor
            The result of passing ``batch`` through the model.

        """
        with set_grad_enabled(not frozen_encoder):
            encoder_out = self.backbone(batch)
        pool_out = self.pool(encoder_out)
        return self.dense_layers(pool_out)


def _conv_config(conv: Conv2d) -> Dict[str, Any]:
    """Return a dictionary with the `conv`'s instantiation arguments.

    Parameters
    ----------
    conv : Conv2d
        Two-dimensional convolutional layer.

    """
    return {
        "in_channels": conv.in_channels,
        "out_channels": conv.out_channels,
        "kernel_size": conv.kernel_size,
        "stride": conv.stride,
        "padding": conv.padding,
        "dilation": conv.dilation,
        "groups": conv.groups,
        "bias": not conv.bias is None,
        "padding_mode": conv.padding_mode,
    }


def _forbidden_args_in_dn_kwargs(user_dn_kwargs: Dict[str, Any]):
    """Check there are no forbidden arguments in ``user_dn_kwargs``.

    Parameters
    ----------
    user_dn_kwargs : Dict[str, Any]
        The dense net kwargs supplied by the user.

    Raises
    ------
    RuntimeError
        If ``in_feats`` is in ``self._dn_args``, the user has tried to
        set the number of input features to the fully connected final
        layer, which is forbidden.
    RuntimeError
        If ``out_feats`` is in ``self._dn_args``, the user has tried to
        set the number of input features to the fully connected final
        layer, which is forbidden.

    """
    if "in_feats" in user_dn_kwargs:
        msg = "Do not supply 'in_feats' in 'fc_net_kwargs'. This "
        msg += "quantity is determined by the choice of encoder."
        raise RuntimeError(msg)
    if "out_feats" in user_dn_kwargs:
        msg = "Do not supply 'out_feats' in 'fc_net_kwargs'. Instead "
        msg += "set this quanitiy using the 'out_feats' argument of "
        msg += "'ConvNet2d' at instantiation."
        raise RuntimeError(msg)
