"""Two-dimensional convolutional encoder moder."""

from torch.nn import Module

from torch import Tensor


class Encoder(Module):
    """Encoder model for image-like inputs."""

    def __init__(self, num_layers: int, features_start: int):
        """Build `Encoder`."""
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
            The result of passing `batch` through the model.

        """
        raise NotImplementedError()
