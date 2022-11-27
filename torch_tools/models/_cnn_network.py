"""CNN classification/regression model."""
from typing import Dict, Any, Optional

from torch import Tensor, set_grad_enabled
from torch.nn import Module, Sequential, Flatten

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
        'avg-max-concat' (the latter simply concatenates the formed two).
        See `torch_tools.models._adaptive_pools_2d` for more info.
    dense_net_kwargs : Dict[str, Any]
        Keyword arguments for `torch_tools.models._dense_network._DenseNetwork`
        which serves as the classification/regression part of the model.

    Notes
    -----
    Because we use torchvision's available architectures, the number of input
    channels needs to be three. A simple workaround for this is to repeat
    greyscale images to have three channels, or wrap the model in a
    `Sequential` and manage the number of channels with another layer.


    """

    def __init__(
        self,
        out_feats: int,
        encoder_style: str = "resnet34",
        pretrained=True,
        pool_style: str = "avg-max-concat",
        dense_net_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Build `ConvNet2d`."""
        super().__init__()
        # TODO: get output_size argument from `get_encoder_backbone`
        self._backbone, num_feats, pool_size = get_backbone(
            encoder_style,
            pretrained=pretrained,
        )

        # TODOL add output_size argument to get_adaptive_pool.
        self._pool = Sequential(
            get_adaptive_pool(pool_style, pool_size),
            Flatten(),
        )

        if dense_net_kwargs is not None:
            self._dn_args.update(dense_net_kwargs)

        self._dense = DenseNetwork(
            2 * num_feats if pool_style == "avg-max-concat" else num_feats,
            out_feats,
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
            encoder_out = self._backbone(batch)
        pool_out = self._pool(encoder_out)
        return self._dense(pool_out)
