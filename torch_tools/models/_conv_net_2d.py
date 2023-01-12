"""CNN classification/regression model."""
from typing import Dict, Any, Optional

from torch import Tensor, set_grad_enabled
from torch.nn import Module, Sequential, Flatten, Conv2d

from torch_tools.models._argument_processing import process_num_feats
from torch_tools.models._encoder_backbones_2d import get_backbone
from torch_tools.models._adaptive_pools_2d import get_adaptive_pool
from torch_tools.models._dense_network import DenseNetwork

# pylint: disable=too-many-arguments


class ConvNet2d(Module):
    """Two-dimensional CNN for image-like objects.

    Parameters
    ----------
    out_feats : int
        The number of output features the model should produce (for example,
        the number of classes).
    in_channels : int
        Number of input channels the model should take. Warning: if you don't
        use three input channels, the first conv layer is overwritten, which
        renders freezing the encoder pointless.
    encoder_option : str
        The encoder option to use. The encoders are loaded from torchvision's
        models. Options include all of torchvision's VGG and ResNET options
        (i.e. 'vgg11', 'vgg11_bn', 'resnet18', etc.).
    pretrained : bool
        Determines whether the encoder is initialised with torchvision's
        pretrained weights. The model will load the most up-to-date
        image-net-trained weights.
    pool_option : str
        The type of adaptive pooling layer to use. Choose from 'avg', 'max' or
        'avg-max-concat' (the latter simply concatenates the former two).
        See `torch_tools.models._adaptive_pools_2d` for more info.
    dense_net_kwargs : Dict[str, Any]
        Keyword arguments for `torch_tools.models._dense_network._DenseNetwork`
        which serves as the classification/regression part of the model.

    """

    def __init__(
        self,
        out_feats: int,
        in_channels: int = 3,
        encoder_style: str = "resnet34",
        pretrained=True,
        pool_style: str = "avg-max-concat",
        dense_net_kwargs: Optional[Dict[str, Any]] = None,
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

        if dense_net_kwargs is not None:
            self._dn_args.update(dense_net_kwargs)

        self.dense_layers = DenseNetwork(
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
        """Pass `batch`" through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs with shape (N, C, H, W), where N is the
            batch-size, C the number of channels and (H, W) the input
            size.
        frozen_encoder : bool, optional
            If `True`, the gradients are disabled in the encoder. If `False`,
            the gradients are enabled in the encoder.

        Returns
        -------
        Tensor
            The result of passing `batch` through the model.

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
