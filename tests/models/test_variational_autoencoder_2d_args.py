"""Test for the arguments of ``VAE2d``."""

import pytest

from torch_tools.models._variational_autoencoder_2d import VAE2d


def test_in_chans_arg_types():
    """Test the types accepted by the ``in_chans`` argument."""
    # Should work with positive int
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16))
    _ = VAE2d(in_chans=2, out_chans=3, input_dims=(16, 16))

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1.0, out_chans=3, input_dims=(16, 16))
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1j, out_chans=3, input_dims=(16, 16))


def test_in_chans_arg_values():
    """Test the values accepted by the ``in_chans`` arg."""
    # Should work with positive int
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16))
    _ = VAE2d(in_chans=2, out_chans=3, input_dims=(16, 16))

    # Should break with ints less than one
    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=0, out_chans=3, input_dims=(16, 16))

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=-1, out_chans=3, input_dims=(16, 16))


def test_out_chans_arg_types():
    """Test the types accepted by the ``out_chans`` argument."""
    # Should work with positive int
    _ = VAE2d(in_chans=1, out_chans=1, input_dims=(16, 16))
    _ = VAE2d(in_chans=2, out_chans=1, input_dims=(16, 16))

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=1.0, input_dims=(16, 16))
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=1j, input_dims=(16, 16))


def test_out_chans_arg_values():
    """Test the values accepted by the ``out_chans`` argument."""
    # Should work with positive odd ints
    _ = VAE2d(in_chans=1, out_chans=1, input_dims=(16, 16))
    _ = VAE2d(in_chans=2, out_chans=3, input_dims=(16, 16))

    # Should break with non-positive ints
    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, out_chans=0, input_dims=(16, 16))
    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, out_chans=-1, input_dims=(16, 16))
    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, out_chans=-2, input_dims=(16, 16))


def test_input_dims_arg_type():
    """Test the types accepted by the ``input_dims`` arg."""
    # Should work with Tuple[int, int]
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16))

    # Should break with non-tuple
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=[16, 16])

    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims={16, 12})


def test_input_dims_arg_length():
    """Test the lengths accepted by the ``input_dims`` arg."""
    # Should work with tuple of length 2
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16))

    # Should break with inputs whose length is not 2
    with pytest.raises(RuntimeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16,))

    with pytest.raises(RuntimeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16, 16))


def test_input_dims_arg_elements_types():
    """Test the types of the contents of ``input_dims``."""
    # Should work with tuple of ints
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16))

    # Should break with tuples containing any non-int
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16.0, 16))

    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16.0))

    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16j, 16j))


def test_input_dims_arg_element_values():
    """Test the values of the contents of ``input_dims``."""
    # Should work with tuple of ints
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16))

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(1, 0))

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(1, 0))

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(-1, -1))


def test_input_dims_too_small_for_number_of_layers():
    """test we raise a value error if ``input_dims`` is too small."""
    # Should break if features are too small
    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(1, 1))


def test_start_features_argument_types():
    """Test the types accepted by the ``start_features`` argument."""
    # Should work with positive ints
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(128, 128), start_features=1)

    # SHould break with non-int
    with pytest.raises(TypeError):
        _ = VAE2d(
            in_chans=1,
            out_chans=3,
            input_dims=(16, 16),
            start_features=1.0,
        )


def test_start_features_argument_values():
    """Test the values accepted by the ``start_features`` argument."""
    # Should work with positive ints
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), start_features=1)

    # Should break with non-positive ints
    with pytest.raises(ValueError):
        _ = VAE2d(
            in_chans=1,
            out_chans=3,
            input_dims=(16, 16),
            start_features=0,
        )

    with pytest.raises(ValueError):
        _ = VAE2d(
            in_chans=1,
            out_chans=3,
            input_dims=(16, 16),
            start_features=-1,
        )


def test_num_layers_argument_types():
    """Test the types accepted by the ``num_layers`` argument."""
    # Should work for ints
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), num_layers=2)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), num_layers=2.0)


def test_num_layers_argument_values():
    """Test the values accepted by the ``num_layers`` argument."""
    # Should work for ints of 2 or more
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), num_layers=2)

    # Should break with ints less than 2
    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), num_layers=1)

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), num_layers=0)


def test_down_pool_argument_types():
    """Test the types accepted by the ``down_pool`` argument."""
    # Should work with str
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), down_pool="max")

    # Should break with non-str
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), down_pool=123)


def test_down_pool_argument_values():
    """Test the values accepted by the ``down_pool`` arg."""
    # Should work with allowed str
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), down_pool="max")
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), down_pool="avg")

    # Should break with any other string
    with pytest.raises(KeyError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), down_pool="Galadriel")


def test_the_bilinear_argument_types():
    """test the types accepted by the ``bilinear`` argument type."""
    # Should work with bool
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), bilinear=True)
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), bilinear=False)

    # Should work with non-bool
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), bilinear=123)


def test_lr_slope_argument_types():
    """Test the types accepted by the ``lr_slope`` argument."""
    # Should work with floats
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), lr_slope=0.0)
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), lr_slope=0.1)

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), lr_slope=0)

    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), lr_slope=1.0j)


def test_kernel_size_argument_type():
    """Test the types accepted by the ``kernel_size`` argument."""
    # Should work with odd, positive ints
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), kernel_size=1)
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), kernel_size=3)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), kernel_size=1j)

    with pytest.raises(TypeError):
        _ = VAE2d(
            in_chans=1,
            out_chans=3,
            input_dims=(16, 16),
            kernel_size=3.0,
        )


def test_kernel_size_argument_values():
    """Test the values accepted by the ``kernel_size`` argument."""
    # Should work with odd, positive ints.
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), kernel_size=1)
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), kernel_size=3)

    # Should break with any non-odd non-positive ints
    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), kernel_size=-1)

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), kernel_size=0)

    with pytest.raises(ValueError):
        _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), kernel_size=2)


def test_max_down_size_argument_types():
    """Test the types accepted by the ``max_down_feats`` argument."""
    # Should work with positive ints or None
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), max_down_feats=64)
    _ = VAE2d(
        in_chans=1,
        out_chans=3,
        input_dims=(16, 16),
        max_down_feats=None,
    )

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = VAE2d(
            in_chans=1,
            out_chans=3,
            input_dims=(16, 16),
            max_down_feats=1.0,
        )

    with pytest.raises(TypeError):
        _ = VAE2d(
            in_chans=1,
            out_chans=3,
            input_dims=(16, 16),
            max_down_feats=True,
        )


def test_max_down_size_argument_values():
    """Test the values accepted by the ``max_down_feats`` argument."""
    # Should work with positive ints or None
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), max_down_feats=64)
    _ = VAE2d(
        in_chans=1,
        out_chans=3,
        input_dims=(16, 16),
        max_down_feats=None,
    )

    # Should break with non-positive ints
    for size in [-2, -1, 0]:
        with pytest.raises(ValueError):
            _ = VAE2d(
                in_chans=1,
                out_chans=3,
                input_dims=(16, 16),
                max_down_feats=size,
            )


def test_min_up_feats_arg_types():
    """Test the types accepted by the ``min_up_feats``."""
    # Should work with positive ints or None
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), min_up_feats=16)
    _ = VAE2d(
        in_chans=1,
        out_chans=3,
        input_dims=(16, 16),
        min_up_feats=None,
    )

    # Should break with non-ints
    for size in [1.0, 2.0j]:
        with pytest.raises(TypeError):
            _ = VAE2d(
                in_chans=1,
                out_chans=3,
                input_dims=(16, 16),
                min_up_feats=size,
            )


def test_min_up_feats_arg_values():
    """Test the values accepted by the ``min_up_feats`` arg."""
    # Should work with positive ints, or None.
    _ = VAE2d(in_chans=1, out_chans=3, input_dims=(16, 16), min_up_feats=16)
    _ = VAE2d(
        in_chans=1,
        out_chans=3,
        input_dims=(16, 16),
        min_up_feats=None,
    )

    # SHould break if not positive int or None
    for size in [-2, -1, 0]:
        with pytest.raises(ValueError):
            _ = VAE2d(
                in_chans=1,
                out_chans=3,
                input_dims=(16, 16),
                min_up_feats=size,
            )


def test_block_style_argument_values():
    """Test the values accepted by the ``block_style`` argument."""
    # Should work with allowed values
    _ = VAE2d(
        in_chans=1,
        out_chans=3,
        input_dims=(16, 16),
        block_style="double_conv",
    )
    _ = VAE2d(
        in_chans=1,
        out_chans=3,
        input_dims=(16, 16),
        block_style="conv_res",
    )

    # Should break with any other values
    with pytest.raises(ValueError):
        _ = VAE2d(
            in_chans=1,
            out_chans=3,
            input_dims=(16, 16),
            block_style=666,
        )


def test_mean_var_nets_argument_style_arg_values():
    """Test the values accepted by the ``"mean_var_nets"`` arg."""
    # Should work with allowed values
    _ = VAE2d(
        in_chans=16,
        out_chans=3,
        mean_var_nets="linear",
        input_dims=(16, 16),
        start_features=16,
        max_down_feats=16,
    )
    _ = VAE2d(
        in_chans=16,
        out_chans=3,
        mean_var_nets="conv",
        input_dims=None,
        start_features=16,
        max_down_feats=16,
    )

    # Should break with other options
    with pytest.raises(ValueError):
        _ = VAE2d(
            in_chans=16,
            out_chans=3,
            mean_var_nets="Gandalf",
            input_dims=(16, 16),
            start_features=16,
            max_down_feats=16,
        )

    # Should break if input_dims is specificed with conv mean var nets
    with pytest.raises(ValueError):
        _ = VAE2d(
            in_chans=16,
            out_chans=3,
            mean_var_nets="conv",
            input_dims=(16, 16),
            start_features=16,
            max_down_feats=16,
        )

    # Should break if input_dims is not specificed with linear mean var nets
    with pytest.raises(ValueError):
        _ = VAE2d(
            in_chans=16,
            out_chans=3,
            mean_var_nets="linear",
            input_dims=None,
            start_features=16,
            max_down_feats=16,
        )


def test_dropout_argument_types():
    """Test the types accepted by the dropout argument."""
    # Should work with floats
    for dropout in [0.0, 0.5]:
        _ = VAE2d(
            in_chans=16,
            out_chans=3,
            mean_var_nets="linear",
            input_dims=(16, 16),
            start_features=16,
            max_down_feats=16,
            dropout=dropout,
        )

    # Should break with any other type
    for dropout in [1, 0.5j]:
        with pytest.raises(TypeError):
            _ = VAE2d(
                in_chans=16,
                out_chans=3,
                mean_var_nets="linear",
                input_dims=(16, 16),
                start_features=16,
                max_down_feats=16,
                dropout=dropout,
            )


def test_dropout_argument_values():
    """Test the values accepted by the dropout argument."""
    # Should work with floats on [0.0, 1.0)
    for dropout in [0.0, 0.999]:
        _ = VAE2d(
            in_chans=16,
            out_chans=3,
            mean_var_nets="linear",
            input_dims=(16, 16),
            start_features=16,
            max_down_feats=16,
            dropout=dropout,
        )

    # Should break with any other type
    for dropout in [-0.001, 1.0, 1.001]:
        with pytest.raises(ValueError):
            _ = VAE2d(
                in_chans=16,
                out_chans=3,
                mean_var_nets="linear",
                input_dims=(16, 16),
                start_features=16,
                max_down_feats=16,
                dropout=dropout,
            )
