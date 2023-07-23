"""Functions for processing arguments to models and blocks."""
from typing import Tuple, Union


def process_num_feats(num_feats: int) -> int:
    """Process argument giving a number of features.

    Parameters
    ----------
    num_feats : int
        Number of input or output features from a model.

    Returns
    -------
    num_feats : int
        See 'Parameters'.

    Raises
    ------
    TypeError
        If `num_feats` is not an integer.
    ValueError
        If `num_feats` is less than one.

    """
    if not isinstance(num_feats, int):
        msg = f"Number of features should be an int. Got '{type(num_feats)}'."
        raise TypeError(msg)
    if num_feats < 1:
        msg = f"Number of features should be 1 or more. Got '{num_feats}'"
        raise ValueError(msg)

    return num_feats


def process_2d_kernel_size(kernel_size: int) -> int:
    """Process kernel size argument for ``torch.nn.Conv2d`` layers.

    Parameters
    ----------
    kernel_size : int
        Length of the square kernel to use in ``torch.nn.Conv2d`` blocks.

    Raises
    ------
    TypeError
        If ``kernel_size`` is not an int.
    ValueError
        If ``kernel_size`` is less than one.
    ValueError
        If ``kernel_size`` is not odd.

    """
    if not isinstance(kernel_size, int):
        msg = f"'kernel_size' should be int, not '{type(kernel_size)}'."
        raise TypeError(msg)

    if kernel_size < 1:
        msg = f"'kernel_size' should be one or more, got '{kernel_size}'."
        raise ValueError(msg)

    if kernel_size % 2 == 0:
        raise ValueError(f"'kernel_size' should be odd. Got '{kernel_size}'.")

    return kernel_size


def process_boolean_arg(bool_arg: bool) -> bool:
    """Process argument which should be a bool.

    Parameters
    ----------
    bool_arg : bool
        Boolean argument.

    Returns
    -------
    bool_arg : bool
        See 'Parameters'.

    Raises
    ------
    TypeError
        If `bool_arg` is not a bool.

    """
    if not isinstance(bool_arg, bool):
        raise TypeError(f"Expected boolean argument. Got '{type(bool_arg)}'.")

    return bool_arg


def process_dropout_prob(prob: float) -> float:
    """Process argument specifying dropout probability.

    Parameters
    ----------
    prob : float
        Dropout probability. Should be on [0.0, 1.0).

    Returns
    -------
    prob : float
        See 'Parameters'.

    Raises
    ------
    TypeError
        If `prob` is not a float.
    ValueError
        If `prob` is not [0.0, 1.0).

    """
    if not isinstance(prob, float):
        raise TypeError(f"Expected float argument. Got '{type(prob)}'.")
    if not 0.0 <= prob < 1.0:
        raise ValueError(f"Prob should be on [0.0, 1.0). Got '{prob}'.")

    return prob


def process_negative_slope_arg(negative_slope: float) -> float:
    """Process argument specifying negative slope in leaky relu.

    Parameters
    ----------
    negative_slope : float
        The negative slope argument for a leaky relu layer.

    Returns
    -------
    float
        `abs(negative_slope)`.

    Raises
    ------
    TypeError
        If `negative_slope` is not a float.

    """
    if not isinstance(negative_slope, float):
        msg = f"negative_slope should be float. Got '{type(negative_slope)}'"
        raise TypeError(msg)

    return abs(negative_slope)


def process_adaptive_pool_output_size_arg(
    output_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Process the output_size argument of adaptive pooling layer.

    Should be compatible with the `output_size` arguments of:
        - `torch.nn.AdaptiveAvgPool2d`
        - `torch.nn.AdaptiveMaxPool2d`

    Parameters
    ----------
    output_size : Tuple[int, int]
        A tuple specifiying the output size the pooling layer should produce.

    Returns
    -------
    output_size : Tuple[int, int]
        See Parameters.

    Raises
    ------
    TypeError
        If `output_size` is not a tuple.
    TypeError
        If `output_size` does not only contain ints.
    RuntimeError
        If `output_size` is not of length 2.
    ValueError
        If any elements of `output_size` are less than 1.

    """
    if not isinstance(output_size, tuple):
        msg = f"output_size arg should be tuple, got '{type(output_size)}'."
        raise TypeError(msg)

    if not all(map(lambda x: isinstance(x, int), output_size)):
        msg = "output_size arg should only contain ints. Got types: "
        msg += f"'{list(map(type, output_size))}'."
        raise TypeError(msg)

    if len(output_size) != 2:
        msg = "output_size arg should have length 2, got "
        msg += f"'{len(output_size)}'."
        raise RuntimeError(msg)

    if not all(map(lambda x: x >= 1, output_size)):
        msg = f"output_size arg's values should be >= 1, got {output_size}."
        raise ValueError(msg)

    return output_size


def process_str_arg(string_arg: str) -> str:
    """Check `string_arg` is a `str` and return it.

    Parameters
    ----------
    string_arg : str
        A string argument.

    Returns
    -------
    string_arg
        See parameters.

    Raises
    ------
    TypeError
        If `string_arg` is not a string.

    """
    if not isinstance(string_arg, str):
        msg = f"Expect string arg to by 'str', got {type(string_arg)}."
        raise TypeError(msg)
    return string_arg


def process_u_architecture_layers(num_layers: int) -> int:
    """Process the number of layers for a U-like architecture.

    Parameters
    ----------
    num_layers : int
        The number of layers requested in the U-like architecture.

    Returns
    -------
    num_layers : int
        See Parameters.

    Raises
    ------
    TypeError
        If `num_layers` is not an int.
    ValueError
        If `num_layers` is less than 2.

    """
    if not isinstance(num_layers, int):
        msg = "Number of layers in U-like architecture should be an int. Got "
        msg += f"'{type(num_layers)}'."
        raise TypeError(msg)
    if num_layers < 2:
        msg = "Number of layers in U-like archiecture should be at least 2. "
        msg += f"Got '{num_layers}'."
        raise ValueError(msg)
    return num_layers


def process_hidden_sizes(
    hidden_sizes: Union[Tuple[int, ...], None]
) -> Union[Tuple[int, ...], None]:
    """Process the `hidden_sizes` argument passed to `DenseNetwork`.

    Parameters
    ----------
    hidden_sizes : Tuple[int, ...] or None
        A tuple of feature sizes.

    Returns
    -------
    hidden_sizes : Tuple[int, ...] or None
        See Parameters.

    Raises
    ------
    TypeError
        If `hidden_sizes` is not a tuple.
    TypeError
        If the elements of `hidden_sizes` are not all ints.
    ValueError
        If any og the elements in `hidden_sizes` are less than one.

    """
    if hidden_sizes is None:
        return hidden_sizes

    if not isinstance(hidden_sizes, tuple):
        msg = f"Hidden sizes should be tuple. Got '{type(hidden_sizes)}'."
        raise TypeError(msg)

    if not all(map(lambda x: isinstance(x, int), hidden_sizes)):
        msg = "All elements in hidden sizes should be int. Got types: "
        msg += f"'{list(map(type, hidden_sizes))}'."
        raise TypeError(msg)

    if any(map(lambda x: x < 1, hidden_sizes)):
        msg = f"Hidden sizes should all be one or more. Got '{hidden_sizes}'."
        raise ValueError(msg)

    return hidden_sizes


def process_input_dims(input_dims: Tuple[int, int]):
    """Process the ``input_dims`` arguments.

    Parameters
    ----------
    input_dims : Tuple[int, int]
        Tuple of two ints.

    Raises
    ------
    TypeError
        If ``input_dims`` is not a tuple.
    RuntimeError
        If ``input_dims`` is not of length two.
    TypeError
        If ``input_dims`` contains anything other than int.
    ValueError
        If ``input_dims`` contains any values less than ones.


    """
    if not isinstance(input_dims, tuple):
        msg = f"Input dims should be tuple, got '{type(input_dims)}'."
        raise TypeError(msg)

    if len(input_dims) != 2:
        msg = f"'input_dims' should have length two, got '{input_dims}'."
        raise RuntimeError(msg)

    if not all(map(lambda x: isinstance(x, int), input_dims)):
        msg = f"'input_dims' should only contain ints, got '{input_dims}'."
        raise TypeError(msg)

    if min(input_dims) < 1:
        msg = f"'input_dims' should contain 1s or more, got '{input_dims}'."
        raise ValueError(msg)

    return input_dims
