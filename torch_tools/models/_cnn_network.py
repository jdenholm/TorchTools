"""CNN classification/regression model."""
from typing import Dict, Any, Optional

from torch import Tensor
from torch.nn import Module, Sequential, Flatten
from torch import set_grad_enabled

from torch_tools.models._encoder_backbones_2d import get_backbone
from torch_tools.models._adaptive_pools_2d import get_adaptive_pool
from torch_tools.models._dense_network import DenseNetwork


class ConvNet2d(Module):
    """Two-dimensional CNN for image-like objects.

    Parameters
    ----------
    out_feats : int
        The number of output features the model should produce.
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
        which serves as the classification or regression part of the model.

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
        self._backbone, num_feats = get_backbone(
            encoder_style,
            pretrained=pretrained,
        )
        self._pool = Sequential(get_adaptive_pool(pool_style), Flatten())

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
            If `True`, the gradients are disabled in the encoder and it is
            set to eval mode. If `False`, the gradients are enabled in the
            encoder.

        Returns
        -------
        Tensor
            The result of passing `batch` through the model.

        """
        with set_grad_enabled(not frozen_encoder):
            encoder_out = self._backbone(batch)
        pool_out = self._pool(encoder_out)
        return self._dense(pool_out)


if __name__ == "__main__":
    model = ConvNet2d(3, dense_net_kwargs={"input_bnorm": True})
    print(model)
