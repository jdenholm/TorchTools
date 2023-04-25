"""Tests for functions in `torch_tools.misc`."""

import pytest


from torch_tools.misc import divides_by_two_check


def test_divides_by_two_check_types():
    """Test the types accepted by the `to_divide` argument."""
    # Should work with ints which can be divided by 2
    divides_by_two_check(10)

    # Should break with non-int
    with pytest.raises(TypeError):
        divides_by_two_check(2.0)
    with pytest.raises(TypeError):
        divides_by_two_check(2j)


def test_divides_by_two_check_values():
    """Test the values accepted by the `to_divide` argument."""
    # Should work with ints which can be divided by 2
    divides_by_two_check(10)

    # Should break with ints of zero or less
    with pytest.raises(ValueError):
        divides_by_two_check(0)
    with pytest.raises(ValueError):
        divides_by_two_check(-1)

    # Should break with positive ints which don't divide by 2
    with pytest.raises(ValueError):
        divides_by_two_check(1)
    with pytest.raises(ValueError):
        divides_by_two_check(3)
    with pytest.raises(ValueError):
        divides_by_two_check(5)
