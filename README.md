# TorchTools
Handy PyTorch utilities and models which should save you time.


## Installation
To clone the repository:
```bash
git clone link-to-be-added
```

To create the necessary conda environment:
```bash
cd /path/to/TorchTools/
conda env create -f requirements.conda.yaml
```

To install TorchTools:
```bash
conda activate torch-tools
pip install -e .
```

To run the tests:
```bash
pytest tests/
```

## Contents


### Dataset


### Models

#### Dense Network
`DenseNetwork` is a simple, perceptron-style, neural network that can be used
for classification and regression. It is super-simple to use and instantiate.

For example:
```python
>>> from torch_tools.models import DenseNetwork

>>> DenseNetwork(in_feats=256,
                 out_feats=2,
                 hidden_sizes=(128, 64),
                 input_bnorm=True,
                 input_dropout=0.1,
                 hidden_dropout=0.25,
                 hidden_bnorm=True,
                 negative_slope=0.2)
DenseNetwork(
  (_input_block): InputBlock(
    (_fwd_seq): Sequential(
      (0): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Dropout(p=0.1, inplace=False)
    )
  )
  (_dense_blocks): Sequential(
    (0): DenseBlock(
      (_fwd_seq): Sequential(
        (0): Linear(in_features=256, out_features=128, bias=True)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Dropout(p=0.25, inplace=False)
        (3): LeakyReLU(negative_slope=0.2)
      )
    )
    (1): DenseBlock(
      (_fwd_seq): Sequential(
        (0): Linear(in_features=128, out_features=64, bias=True)
        (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Dropout(p=0.25, inplace=False)
        (3): LeakyReLU(negative_slope=0.2)
      )
    )
    (2): DenseBlock(
      (_fwd_seq): Sequential(
        (0): Linear(in_features=64, out_features=2, bias=True)
      )
    )
  )
)
```