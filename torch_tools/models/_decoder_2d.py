"""Two-dimensional decoder model."""

from torch.nn import Module, Sequential

from torch import Tensor

from torch_tools.models._blocks_2d import UpBlock
from torch_tools.models._argument_processing import process_num_feats


class Decoder2d(Module):
    """Simple decoder model for image-like inputs."""

    def __init__(
        self,
        in_chans: int,
        num_blocks: int,
        bilinear: bool,
        lr_slope: float,
    ):
        """Build `Decoder`."""
        super().__init__()
        self._fwd = self._get_blocks(in_chans, num_blocks, bilinear, lr_slope)

    def _get_blocks(
        self,
        in_chans: int,
        num_blocks: int,
        bilinear: bool,
        lr_slope: float,
    ) -> Sequential:
        """Get the upsampling blocks in a `Sequential`."""
        chans = in_chans
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                UpBlock(
                    process_num_feats(chans),
                    process_num_feats(chans // 2),
                    bilinear,
                    lr_slope,
                )
            )
            chans //= 2
        return Sequential(*blocks)

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
        return self._fwd(batch)
