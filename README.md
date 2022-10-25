# TorchTools
Handy PyTorch  datasets and models which should save you time.


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
conda activate torch-tools
pytest tests/
```

## Contents


### Dataset


### Models

#### Dense Network
`DenseNetwork` is a simple, perceptron-style, neural network that can be used
for classification and regression. It is super-simple to use and instantiate.
See `demos/...`.