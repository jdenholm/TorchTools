"""Test for the arguments of ``VAE2d``."""
import pytest

from torch_tools.models._variational_autoencoder_2d import VAE2d


def test_in_chans_arg_types():
    """Test the types accepted by the ``in_chans`` argument."""
    # Should work with positive int
    _ = VAE2d(in_chans=1, input_dims=(64, 64))
    _ = VAE2d(in_chans=2, input_dims=(64, 64))

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1.0, input_dims=(64, 64))
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1j, input_dims=(64, 64))


def test_in_chans_arg_values():
    """Test the values accepted by the ``in_chans`` arg."""
    # Should work with positive int
    _ = VAE2d(in_chans=1, input_dims=(64, 64))
    _ = VAE2d(in_chans=2, input_dims=(64, 64))

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=0, input_dims=(64, 64))

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=-1, input_dims=(64, 64))


def test_input_dims_arg_type():
    """Test the types accepted by the ``input_dims`` arg."""
    # Should work with Tuple[int, int]
    _ = VAE2d(in_chans=1, input_dims=(64, 64))

    # Should break with non-tuple
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, input_dims=[64, 64])

    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, input_dims={64, 12})


def test_input_dims_arg_length():
    """Test the lengths accepted by the ``input_dims`` arg."""
    # Should work with tuple of length 2
    _ = VAE2d(in_chans=1, input_dims=(64, 64))

    # Should break with inputs whose length is not 2
    with pytest.raises(RuntimeError):
        _ = VAE2d(in_chans=1, input_dims=(64,))

    with pytest.raises(RuntimeError):
        _ = VAE2d(in_chans=1, input_dims=(64, 64, 64))


def test_input_dims_arg_elements_types():
    """Test the types of the contents of ``input_dims``."""
    # Should work with tuple of ints
    _ = VAE2d(in_chans=1, input_dims=(64, 64))

    # Should break with tuples containing any non-int
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, input_dims=(64.0, 64))

    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, input_dims=(64, 64.0))

    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, input_dims=(64j, 32j))


def test_input_dims_arg_element_values():
    """Test the values of the contents of ``input_dims``."""
    # Should work with tuple of ints
    _ = VAE2d(in_chans=1, input_dims=(32, 32))

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, input_dims=(1, 0))

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, input_dims=(1, 0))

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, input_dims=(-1, -1))


def test_input_dims_too_small_for_number_of_layers():
    """test we raise a value error if ``input_dims`` is too small."""
    # Should break if features are too small
    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, input_dims=(1, 1))


def test_start_features_argument_types():
    """Test the types accepted by the ``start_features`` argument."""
    # Should work with positive ints
    _ = VAE2d(in_chans=1, input_dims=(128, 128), start_features=1)

    # SHould break with non-int
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, input_dims=(32, 32), start_features=1.0)


def test_start_features_argument_values():
    """Test the values accepted by the ``start_features`` argument."""
    # Should work with positive ints
    _ = VAE2d(in_chans=1, input_dims=(32, 32), start_features=1)

    # Should break with non-positive ints
    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, input_dims=(32, 32), start_features=0)

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, input_dims=(32, 32), start_features=-1)


def test_num_layers_argument_types():
    """Test the types accepted by the ``num_layers`` argument."""
    # Should work for ints
    _ = VAE2d(in_chans=1, input_dims=(32, 32), num_layers=2)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, input_dims=(32, 32), num_layers=2.0)


def test_num_layers_argument_values():
    """Test the values accepted by the ``num_layers`` argument."""
    # Should work for ints of 2 or more
    _ = VAE2d(in_chans=1, input_dims=(32, 32), num_layers=2)

    # Should break with ints less than 2
    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, input_dims=(32, 32), num_layers=1)

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, input_dims=(32, 32), num_layers=0)


def test_down_pool_argument_types():
    """Test the types accepted by the ``down_pool`` argument."""
    # Should work with str
    _ = VAE2d(in_chans=1, input_dims=(32, 32), down_pool="max")

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, input_dims=(32, 32), down_pool=123)


def test_down_pool_argument_values():
    """Test the values accepted by the ``down_pool`` arg."""
    # Should work with allowed str
    _ = VAE2d(in_chans=1, input_dims=(32, 32), down_pool="max")
    _ = VAE2d(in_chans=1, input_dims=(32, 32), down_pool="avg")

    # Should break with any other string
    with pytest.raises(KeyError):
        _ = VAE2d(in_chans=1, input_dims=(32, 32), down_pool="Galadriel")


def test_the_bilinear_argument_types():
    """test the types accepted by the ``bilinear`` argument type."""
    # Should work with bool
    _ = VAE2d(in_chans=1, input_dims=(32, 32), bilinear=True)
    _ = VAE2d(in_chans=1, input_dims=(32, 32), bilinear=False)

    # Should work with non-bool
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, input_dims=(32, 32), bilinear=123)


def test_lr_slope_argument_types():
    """Test the types accepted by the ``lr_slope`` argument."""
    # Should work with floats
    _ = VAE2d(in_chans=1, input_dims=(32, 32), lr_slope=0.0)
    _ = VAE2d(in_chans=1, input_dims=(32, 32), lr_slope=0.1)

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, input_dims=(32, 32), lr_slope=0)

    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, input_dims=(32, 32), lr_slope=1.0j)
