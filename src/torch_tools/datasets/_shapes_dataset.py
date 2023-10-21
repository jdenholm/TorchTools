"""Synthetic dataset object."""
from typing import Tuple

from torch import from_numpy, Tensor  # pylint: disable=no-name-in-module
from torch.utils.data import Dataset

from numpy import ones, float32, ndarray, array
from numpy.random import default_rng

from skimage.draw import disk, rectangle  # pylint: disable=no-name-in-module


class ShapesDataset(Dataset):
    """Synthetic shape dataset.

    Parameters
    ----------
    spot_prob : float, optional
        Probability of including spots in the image.
    square_prob : float, optional
        Probability of including sqaures in the image.
    num_spots : int, optional
        The number of spots that will be included in the image.
    num_squares : int, optional
        The number of squares that will be included in the image.
    length : int, optional
        The length of the data set.
    image_size : int, optional
        The length of the square images.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        spots_prob: float = 0.5,
        square_prob: float = 0.5,
        num_spots: int = 10,
        num_squares: int = 10,
        length: int = 1000,
        image_size: int = 256,
    ):
        """Build ``ShapesDataset``."""
        self._len = length
        self._spot_prob = spots_prob
        self._square_prob = square_prob
        self._num_spots = num_spots
        self._num_squares = num_squares
        self._img_size = image_size

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

        Returns
        -------
        Tensor
            An RGB image of shape (c, H, W).
        Tensor
            Target vector:

                — If there are no spots or squares, [0.0, 0.0]
                — If there are spots only, [1.0, 0.0]
                — If there are squares only, [0.0, 1.0]
                — If there are both, [1.0, 1.0]

        """
        return self._create_image()

    def _add_spots(self, image: ndarray) -> bool:
        """Add spots to ``image``.

        Parameters
        ----------
        image : ndarray
            RGB image.

        Returns
        -------
        include_spots : bool
            Whether or not the spots were added.

        """
        include_spots = self._rng.random() <= self._spot_prob

        if include_spots:
            for _ in range(self._num_spots):
                colour = self._rng.random(size=3)
                radius = self._img_size // 20
                centre = self._rng.integers(
                    radius,
                    self._img_size - radius,
                    size=2,
                )

                rows, cols = disk(centre, radius)

                image[rows, cols, :] = colour

        return include_spots

    def _add_squares(self, image: ndarray) -> bool:
        """Add spots to ``image``.

        Parameters
        ----------
        image : ndarray
            RGB image.

        Returns
        -------
        include_squares : bool
            Whether or no squares were included.

        """
        include_squares = self._rng.random() <= self._square_prob

        if include_squares:
            for _ in range(self._num_spots):
                colour = self._rng.random(size=3)
                length = self._img_size // 20
                origin = self._rng.integers(
                    0,
                    self._img_size - (2 * length),
                    size=2,
                )

                rows, cols = rectangle(origin, origin + (2 * length))

                image[rows, cols, :] = colour

        return include_squares

    def _create_image(self) -> Tuple[Tensor, Tensor]:
        """Create image.

        Returns
        -------
        Tensor
            An RGB image of shape (c, H, W).
        Tensor
            Target vector:

                — If there are no spots or squares, [0.0, 0.0]
                — If there are spots only, [1.0, 0.0]
                — If there are squares only, [0.0, 1.0]
                — If there are both, [1.0, 1.0]

        """
        image = ones((self._img_size, self._img_size, 3), dtype=float32)

        spots = self._add_spots(image)
        squares = self._add_squares(image)

        return (
            from_numpy(image).permute(2, 0, 1),
            from_numpy(array([spots, squares])).float(),
        )
