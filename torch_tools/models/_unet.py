"""UNet model for semantic segmentation."""

from torch import Tensor

from torch.nn import Module


class UNet(Module):
    """UNet model for semantic segmentation."""

    def __init__(self):
        """Build `UNet`."""
        super().__init__()

    def forward(self, batch: Tensor) -> Tensor:
        """Pass `batch` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        Returns
        -------
        Tensor
            The resul of passing batch through the model.

        """
        raise NotImplementedError("UNet forward method not implemented.")
