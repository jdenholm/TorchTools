"""Tests for functions in `torch_tools.models._argument_processing`."""
import pytest


from torch_tools.models import _argument_processing as ap


def test_process_num_feats_types():
    """Test `ap.process_num_feats`'s type checking."""
    # Should work with int
    _ = ap.process_num_feats(1)

    # Should break with any non-int
    with pytest.raises(TypeError):
        _ = ap.process_num_feats(1.0)
    with pytest.raises(TypeError):
        _ = ap.process_num_feats("1.0")
    with pytest.raises(TypeError):
        _ = ap.process_num_feats(1.0j)


def test_process_num_feats_values():
    """Test `ap.process_num_feats`'s value checking."""
    # Should work with ints of one and above
    _ = ap.process_num_feats(1)
    _ = ap.process_num_feats(2)

    # Should fail with anything less than one
    with pytest.raises(ValueError):
        _ = ap.process_num_feats(0)
    with pytest.raises(ValueError):
        _ = ap.process_num_feats(-1)


def test_process_2d_kernel_size_arg_types():
    """Test the types accepted by the ``kernel_size`` argument."""
    # Should work with odd, popsitive ints.
    _ = ap.process_2d_kernel_size(1)
    _ = ap.process_2d_kernel_size(3)
    _ = ap.process_2d_kernel_size(5)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = ap.process_2d_kernel_size(1.0)

    with pytest.raises(TypeError):
        _ = ap.process_2d_kernel_size(1j)


def test_process_2d_kernel_size_arg_values():
    """Test the values accepted by the ``kernel_size`` argument."""
    # Should work with odd, popsitive ints.
    _ = ap.process_2d_kernel_size(1)
    _ = ap.process_2d_kernel_size(3)
    _ = ap.process_2d_kernel_size(5)

    # Should break with ints less than on
    with pytest.raises(ValueError):
        _ = ap.process_2d_kernel_size(0)

    with pytest.raises(ValueError):
        _ = ap.process_2d_kernel_size(-1)

    with pytest.raises(ValueError):
        _ = ap.process_2d_kernel_size(-2)

    # Should break with even positive ints
    with pytest.raises(ValueError):
        _ = ap.process_2d_kernel_size(2)

    with pytest.raises(ValueError):
        _ = ap.process_2d_kernel_size(4)

    with pytest.raises(ValueError):
        _ = ap.process_2d_kernel_size(6)


def test_process_boolean_arg_types():
    """Test `ap.process_boolean_arg`'s type checking."""
    # Should work with bool
    _ = ap.process_boolean_arg(True)
    _ = ap.process_boolean_arg(False)

    # Should fail with non-bool
    with pytest.raises(TypeError):
        _ = ap.process_boolean_arg(1)
    with pytest.raises(TypeError):
        _ = ap.process_boolean_arg(1.0)
    with pytest.raises(TypeError):
        _ = ap.process_boolean_arg("Batman")


def test_process_dropout_argument_types():
    """Test `ap.process_dropout_argument`'s type checking."""
    # Should work with floats on (0.0, 1.0]
    _ = ap.process_dropout_prob(0.0)
    _ = ap.process_dropout_prob(0.99)

    # Should fail any non-floats
    with pytest.raises(TypeError):
        _ = ap.process_dropout_prob(0)
    with pytest.raises(TypeError):
        _ = ap.process_dropout_prob("Hello")
    with pytest.raises(TypeError):
        _ = ap.process_dropout_prob(1.0j)


def test_process_dropout_argument_values():
    """Test `ap.process_dropout_argument`'s value checking."""
    # Should work with floats on [0.0, 1.0)
    _ = ap.process_dropout_prob(0.0)
    _ = ap.process_dropout_prob(0.5)
    _ = ap.process_dropout_prob(0.99)

    # Should fail with floats outwith [0.0,, 1.0)
    with pytest.raises(ValueError):
        _ = ap.process_dropout_prob(-0.00001)
    with pytest.raises(ValueError):
        _ = ap.process_dropout_prob(1.0)
    with pytest.raises(ValueError):
        _ = ap.process_dropout_prob(1.0001)


def test_negative_slope_argument_type():
    """Test `ap.process_negative_slope_argument`'s type checking."""
    # Should work with floats
    _ = ap.process_negative_slope_arg(-1.0)
    _ = ap.process_negative_slope_arg(0.0)
    _ = ap.process_negative_slope_arg(1.0)

    # Should break with non-float
    with pytest.raises(TypeError):
        _ = ap.process_negative_slope_arg(1)
    with pytest.raises(TypeError):
        _ = ap.process_negative_slope_arg(1j)
    with pytest.raises(TypeError):
        _ = ap.process_negative_slope_arg("I'm batman.")


def test_adaptive_pool_output_size_arg_type():
    """Test the argument type of `ap.process_adaptive_pool_output_size_arg`."""
    # Should work with Tuple[int, int] where the ints are one or more.
    ap.process_adaptive_pool_output_size_arg((1, 1))

    # Should break with non-tuple.
    with pytest.raises(TypeError):
        ap.process_adaptive_pool_output_size_arg([1, 1])


def test_adaptive_pool_output_size_arg_contents_types():
    """Test the types in output-size arg.

    Makes sure the the `output_size` arg of
    `ap.process_adaptive_pool_output_size_arg` can only be ints.

    """
    # Should work with Tuple[int, int] where the ints are one or more.
    ap.process_adaptive_pool_output_size_arg((1, 1))

    # Should break if the tuple is not all ints.
    with pytest.raises(TypeError):
        ap.process_adaptive_pool_output_size_arg([1.0, 1])
    with pytest.raises(TypeError):
        ap.process_adaptive_pool_output_size_arg([1, 1.0])
    with pytest.raises(TypeError):
        ap.process_adaptive_pool_output_size_arg([1j, 1j])


def test_adaptive_pool_output_size_arg_length():
    """Test the length og the output_size arg must be 2.

    Makes sure the length of the `output_size` arg of
    `ap.process_adaptive_pool_output_size_arg` must have length 2.

    """
    # Should work with Tuple[int, int] where the ints are one or more.
    ap.process_adaptive_pool_output_size_arg((1, 1))

    # Should break if the tuple has more, or fewer, than two elements.
    with pytest.raises(RuntimeError):
        ap.process_adaptive_pool_output_size_arg((1,))
    with pytest.raises(RuntimeError):
        ap.process_adaptive_pool_output_size_arg((1, 1, 1))


def test_adaptice_pool_output_size_arg_contents_vvalues():
    """Test the values passed inside `output_size` arg.

    Makes sure the values if the `output_size` arg of
    `ap.process_adaptive_pool_output_size_arg` are one or more.

    """
    # Should work with Tuple[int, int] where the ints are one or more.
    ap.process_adaptive_pool_output_size_arg((1, 1))

    # Should break if any of the ints are less than one.
    with pytest.raises(ValueError):
        ap.process_adaptive_pool_output_size_arg((1, 0))
    with pytest.raises(ValueError):
        ap.process_adaptive_pool_output_size_arg((0, 1))
    with pytest.raises(ValueError):
        ap.process_adaptive_pool_output_size_arg((0, 0))


def test_process_str_arg_types():
    """Test `process_str_arg` catches non-str."""
    # Should work with strings
    ap.process_str_arg("Glorfindel")

    # Should break with non-str
    with pytest.raises(TypeError):
        ap.process_str_arg(123)
    with pytest.raises(TypeError):
        ap.process_str_arg({})


def test_process_u_architecture_layers_types():
    """Test the types accepted by `process_u_architecture_layers`."""
    # Should work with ints of 2 or more
    _ = ap.process_u_architecture_layers(2)

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = ap.process_u_architecture_layers(2.0)
    with pytest.raises(TypeError):
        _ = ap.process_u_architecture_layers(2j)


def test_process_u_architecture_layers_values():
    """Test the values accepted by `process_u_architecture_layers`."""
    # Should work with ints of 2 or more
    _ = ap.process_u_architecture_layers(2)
    _ = ap.process_u_architecture_layers(3)
    _ = ap.process_u_architecture_layers(4)

    # Should break with ints less than two
    with pytest.raises(ValueError):
        _ = ap.process_u_architecture_layers(1)
    with pytest.raises(ValueError):
        _ = ap.process_u_architecture_layers(0)


def test_process_hidden_sizes_argument_types():
    """Test the types accepted by `ap.process_hidden_sizes`."""
    # Should work with Tuple[int, ...] or None (ints should be 1 or more)
    ap.process_hidden_sizes((10, 10, 1))
    ap.process_hidden_sizes(None)

    # Should break with non-tuple or non-None
    with pytest.raises(TypeError):
        ap.process_hidden_sizes([10, 10, 1])
    with pytest.raises(TypeError):
        ap.process_hidden_sizes("Elrond of Rivendell.")


def test_process_hidden_sizes_argument_individual_size_types():
    """Test the types of the individual sizes in the `hidden_sizes` arg."""
    # Should work with Tuple[int, ...]
    ap.process_hidden_sizes((10, 10, 1))

    # Should break if there are any non-ints in the tuple
    with pytest.raises(TypeError):
        ap.process_hidden_sizes((10, 10, 1.0))
    with pytest.raises(TypeError):
        ap.process_hidden_sizes((10j, 10, 1))
    with pytest.raises(TypeError):
        ap.process_hidden_sizes((10, "Peregrin Took", 1))


def test_process_hidden_sizes_argument_values():
    """Test the values accepted by the `hidden_sizes` argument."""
    # Should work with Tuple[int, ...] or None (with ints of 1 or more)
    ap.process_hidden_sizes((10, 10, 1))

    # Should break if any of the ints are less than 1
    with pytest.raises(ValueError):
        ap.process_hidden_sizes((10, 10, -1))
    with pytest.raises(ValueError):
        ap.process_hidden_sizes((0, 10, 1))


def test_process_input_dims_arg_types():
    """Test ``process_input_dims`` argument type."""
    # Should work with Tuple[int, int]
    _ = ap.process_input_dims((10, 10))

    # Should break with non-tuple
    with pytest.raises(TypeError):
        _ = ap.process_input_dims([1, 2])

    with pytest.raises(TypeError):
        _ = ap.process_input_dims({1, 2})


def test_process_input_dims_arg_length():
    """Test ``process_input_dims`` argument length."""
    # Should work with tuple of length 2
    _ = ap.process_input_dims((10, 10))

    # Should break with tuples whose length are not 2
    with pytest.raises(RuntimeError):
        _ = ap.process_input_dims((10,))

    with pytest.raises(RuntimeError):
        _ = ap.process_input_dims((10, 10, 10))


def test_process_input_dims_arg_element_types():
    """Test the types allowed in the ``input_dims`` arg."""
    # Should work with tuples containing ints
    _ = ap.process_input_dims((10, 10))

    # Should break with non-ints
    with pytest.raises(TypeError):
        _ = ap.process_input_dims((10, 10.0))

    with pytest.raises(TypeError):
        _ = ap.process_input_dims((10.0, 10))

    with pytest.raises(TypeError):
        _ = ap.process_input_dims((10, 10j))


def test_process_input_dims_element_values():
    """Test the values allowed in the ``input_dims`` arg."""
    # Should work with positive ints
    _ = ap.process_input_dims((1, 1))
    _ = ap.process_input_dims((2, 2))

    # Should break with non-positve ints
    with pytest.raises(ValueError):
        _ = ap.process_input_dims((1, 0))

    with pytest.raises(ValueError):
        _ = ap.process_input_dims((1, 0))

    with pytest.raises(ValueError):
        _ = ap.process_input_dims((-1, -1))


def test_process_optional_features_arg_types():
    """Test the types accepted by the ``max_feats`` argument."""
    # Should work with ints or None
    _ = ap.process_optional_feats_arg(max_feats=1)
    _ = ap.process_optional_feats_arg(max_feats=None)

    # Should break with non-int
    with pytest.raises(TypeError):
        _ = ap.process_optional_feats_arg(max_feats=1.0)

    with pytest.raises(TypeError):
        _ = ap.process_optional_feats_arg(max_feats=1.0j)


def test_process_optional_features_arg_values():
    """Test the values accepted by the ``max_feats`` argument."""
    # Should work with ints of one or more, or None
    _ = ap.process_optional_feats_arg(max_feats=1)
    _ = ap.process_optional_feats_arg(max_feats=None)

    # Should break with ints less than 1
    with pytest.raises(ValueError):
        _ = ap.process_optional_feats_arg(max_feats=0)

    with pytest.raises(ValueError):
        _ = ap.process_optional_feats_arg(max_feats=-1)
