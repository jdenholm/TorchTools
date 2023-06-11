"""Miscellaneous helpful functions."""


def divides_by_two_check(to_divide: int):
    """Make sure ``to_divide`` can be divided by 2.

    Parameters
    ----------
    to_divide : int
        A number to be divided by two.

    Raises
    ------
    TypeError
        If ``to_divide`` is not an int.
    ValueError
        If ``to_divide`` is not greater than zero.
    ValueError
        If ``to_divide / 2`` is irrational.

    """
    if not isinstance(to_divide, int):
        raise TypeError(f"'to_divide' should be in. Got '{type(to_divide)}'.")
    if to_divide <= 0:
        msg = f"'to_divide' should be greater than zero. Got '{to_divide}'"
        raise ValueError(msg)
    if (to_divide % 2) != 0:
        raise ValueError(f"'{to_divide}' does not divide by 2.")
