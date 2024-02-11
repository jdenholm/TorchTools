"""Weight initialisation functions."""

from torch.nn import Module, Parameter
from torch import randn_like, no_grad  # pylint: disable=no-name-in-module


def normal_init(
    model: Module,
    attr_name: str = "weight",
    mean: float = 0.0,
    std: float = 0.02,
):
    """Initialise ``model``'s weights by sampling from a normal distribution.

    The weights *and* biases are initialised.

    Parameters
    ----------
    model : Module
        The ``Module`` to be initialised.
    attr_name : str
        The name of the attriubute in ``model`` to be initialised with
        normally distributed data.
    mean : float, optional
        The mean of the normal distribution the weights are sampled from.
    std : float, optional
        The standard deviation of the normal distribution the weights are
        sampled from.

    Raises
    ------
    TypeError
        If ``model`` is not an instance of ``torch.nn.Module``.
    TypeError
        If ``attr_name`` is not a str.
    TypeError
        If ``mean`` is not a float.
    TypeError
        If ``std`` is not a float.

    """
    if not isinstance(model, Module):
        msg = "Weight init can only be applied to torch.nn.Module. "
        msg += f"Got '{type(model)}'."
        raise TypeError(msg)

    if not isinstance(attr_name, str):
        raise TypeError(f"'attr_name' should be str. Got '{type(attr_name)}'.")

    if not isinstance(mean, float):
        raise TypeError(f"'mean' should be a float. Got '{type(mean)}'.")

    if not isinstance(std, float):
        raise TypeError(f"'std' should be a float. Got '{std}'.")

    if hasattr(model, attr_name):
        with no_grad():
            attr = getattr(model, attr_name)
            if attr is None:
                set_to = None
            else:
                set_to = (
                    Parameter((randn_like(attr) * std) + mean)
                    if attr is not None
                    else None
                )

            setattr(model, attr_name, set_to)
