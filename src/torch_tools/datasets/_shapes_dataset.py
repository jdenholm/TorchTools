"""Synthetic dataset object."""

from typing import Tuple, Optional

from torch import from_numpy, Tensor  # pylint: disable=no-name-in-module
from torch.utils.data import Dataset

from torchvision.transforms import Compose  # type: ignore

from numpy import ndarray, array, where, full
from numpy.random import default_rng

from skimage.morphology import star, square, octagon, disk


class ShapesDataset(Dataset):
    """Synthetic dataset which produces images withs spots and squares.

    *Warning—*this dataset object is untested.

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
    input_tfms : Compose, optional
        A composition of transforms to apply to the input.
    target_tfms : Compose, optional
        A composition of transforms to apply to the target.
    seed : int
        Integer seed for numpy's default rng.

    Notes
    -----
    The images have white backrounds and the shapes have randomly selected
    RGB colours on [0, 1)^{3}.


    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        spot_prob: float = 0.5,
        square_prob: float = 0.5,
        star_prob: float = 0.5,
        octagon_prob: float = 0.5,
        num_shapes: int = 3,
        length: int = 1000,
        image_size: int = 256,
        input_tfms: Optional[Compose] = None,
        target_tfms: Optional[Compose] = None,
        seed: int = 666,
    ):
        """Build ``ShapesDataset``."""
        self._len = length
        self._num_shapes = num_shapes

        self._probs = {
            "square": square_prob,
            "spot": spot_prob,
            "star": star_prob,
            "octagon": octagon_prob,
        }

        self._img_size = image_size
        self._x_tfms = input_tfms
        self._y_tfms = target_tfms

        self._rng = default_rng(seed=seed)

    _shapes = {
        "square": lambda x: square(2 * x),
        "star": star,
        "octagon": lambda x: octagon(x, x),
        "spot": disk,
    }

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
        img : Tensor
            An RGB image of shape (c, H, W).
        tgt : Tensor
            Target vector:

                — If there are no spots or squares, [0.0, 0.0]
                — If there are spots only, [1.0, 0.0]
                — If there are squares only, [0.0, 1.0]
                — If there are both, [1.0, 1.0]

        """
        img, tgt = self._create_image()

        if self._x_tfms is not None:
            img = self._x_tfms(img)

        if self._y_tfms is not None:
            tgt = self._y_tfms(tgt)

        return img, tgt

    def _add_shape(self, image: ndarray, shape: str) -> bool:
        """Add spots to ``image``.

        Parameters
        ----------
        image : ndarray
            RGB image.
        shape : str
            Name of the shape to include.


        Returns
        -------
        include_spots : bool
            Whether or not the spots were added.

        """
        include_shape = self._rng.random() <= self._probs[shape]

        if include_shape:
            for _ in range(self._num_shapes):
                colour = self._rng.random(size=(1, 3))
                radius = self._img_size // 20
                shape_arr = self._shapes[shape](radius)

                # pylint: disable=unbalanced-tuple-unpacking
                rows, cols = where(shape_arr == 1)
                left, top = self._rng.integers(
                    0,
                    self._img_size - len(shape_arr),
                    size=2,
                )

                rows, cols = rows + top, cols + left

                image[rows, cols] = colour

        return include_shape

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
        # image = ones((self._img_size, self._img_size, 3), dtype=float32)

        image = full(
            (self._img_size, self._img_size, 3),
            fill_value=self._rng.random(size=(1, 3)),
        )

        targets = []
        for key in self._shapes:
            targets.append(self._add_shape(image, key))

        return (
            from_numpy(image).permute(2, 0, 1).float(),
            from_numpy(array(targets)).float(),
        )
