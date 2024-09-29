"""Test the call behaviour of ``VAE2d``."""

from itertools import product

from torch import rand, no_grad  # pylint: disable=no-name-in-module

from torch_tools.models._variational_autoencoder_2d import VAE2d


def test_vae_call_return_shapes():  # pylint: disable=too-many-locals
    """Test the return types in the call method of ``VAE2d``."""
    in_channels = [1, 3]
    out_channels = [1, 3]
    num_layers = [3]
    features_start = [8, 16]
    slopes = [0.0, 0.1]
    pools = ["avg", "max"]
    bilinear = [True, False]
    kernel_size = [1, 3]
    image_dims = [(8, 16), (16, 32)]
    max_features = [32, None]
    min_features = [None, 16]
    block_styles = ["double_conv", "conv_res"]
    mean_vars = ["linear", "conv"]
    dropout = [0.0, 0.666]

    iterator = product(
        in_channels,
        out_channels,
        features_start,
        num_layers,
        pools,
        bilinear,
        slopes,
        kernel_size,
        image_dims,
        max_features,
        min_features,
        block_styles,
        mean_vars,
        dropout,
    )

    for (
        in_chans,
        out_chans,
        start_feats,
        layers,
        pool,
        bilin,
        slope,
        k_size,
        in_dims,
        max_feats,
        min_feats,
        block,
        mv_net,
        drop,
    ) in iterator:

        model = VAE2d(
            in_chans=in_chans,
            out_chans=out_chans,
            input_dims=in_dims if mv_net == "linear" else None,
            start_features=start_feats,
            num_layers=layers,
            down_pool=pool,
            bilinear=bilin,
            lr_slope=slope,
            kernel_size=k_size,
            max_down_feats=max_feats,
            min_up_feats=min_feats,
            block_style=block,
            mean_var_nets=mv_net,
            dropout=drop,
        )

        batch = rand(1, in_chans, *in_dims)

        model.train()
        with no_grad():
            preds, kl_div = model(batch)
            assert preds.shape == (1, out_chans) + in_dims
            assert kl_div.numel() == 1

        model.eval()
        with no_grad():
            preds, kl_div = model(batch)
            assert preds.shape == (1, out_chans) + in_dims
            assert kl_div.numel() == 1
