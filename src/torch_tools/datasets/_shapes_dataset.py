"""Synthetic dataset object."""

from torch.utils.data import Dataset

from numpy.random import default_rng


class ShapesDataset(Dataset):
    """Synthetic shapes dataset."""

    def __init__(
        self,
        spots_prob: float,
        square_prob: float,
        length: int = 1000,
        image_size: int = 256,
    ):
        """Build ``ShapesDataset``."""
        self._len = length

    _rng = default_rng(seed=123)

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self._len

    def __getitem__(self, idx: int):
        """Return an image-target pair.

        Parameters
        ----------
        idx : int
            The index of the item to be returned.

        """
