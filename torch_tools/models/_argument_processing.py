"""Functions for processing arguments to models and blocks."""


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


def process_boolean_arg(bool_arg: bool) -> bool:
    """Process argument which should be a bool.

    Parameters
    ----------
    bool_arg : bool
        Boolean argument.

    Returns
    ----------
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
        Dropout probability.

    Returns
    -------
    prob : float
        See 'Parameters'.

    Raises
    ------
    TypeError
        If `prob` is not a float.
    ValueError
        If `prob` is not (0.0, 1.0].

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

    """
    if not isinstance(negative_slope, float):
        msg = f"negative_slope should be float. Got '{type(negative_slope)}'"
        raise TypeError(msg)
    return abs(negative_slope)
