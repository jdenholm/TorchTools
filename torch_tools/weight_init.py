"""Weight initialisation functions."""

from torch.nn import Module, init, Conv1d, Conv2d, Conv3d, Linear


def normal_init(model: Module, mean: float = 0.0, std: float = 0.02):
    """Initialise ``model``'s weights by sampling from a normal distribution.

    The weights *and* biases are initialised.

    Parameters
    ----------
    model : Module
        The ``Module`` to be initialised.
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
        If ``mean`` is not a float.
    TypeError
        If ``std`` is not a float.

    """
    if not isinstance(model, Module):
        msg = "Weight init can only be applied to torch.nn.Module. "
        msg += f"Got '{type(model)}'."
        raise TypeError(msg)
    if not isinstance(mean, float):
        raise TypeError(f"'mean' should be a float. Got '{type(mean)}'.")
    if not isinstance(std, float):
        raise TypeError(f"'std' should be a float. Got '{std}'.")

    if isinstance(model, (Conv1d, Conv2d, Conv3d, Linear)):
        init.normal_(model.weight, mean=mean, std=std)
        if model.bias is not None:
            init.normal_(model.bias, mean=mean, std=std)
