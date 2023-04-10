# TorchTools
Handy PyTorch models and utilities which should save you time.

Please see [**the documentation**](https://jdenholm.github.io/TorchTools/).


## Description
This Python library contains a bunch of neural networks, implemented in PyTorch, and a generic ``Dataset`` class with useful features to help you get started with your deep-learning problem more quickly, and spend less time engineering and testing—because researchers *always* test their code—the PyTorch-specific software you need. There is also the hope that you will write and duplicate less code by being able to easily set and modify these models' architectures at instantiation.


## Authors and Acknowledgement
This project was created by Jim Denholm. Any contributions, criticisms or suggestions are welcome.


## License
Distributed under the MIT License. See LICENSE.md for more information.




## Getting Started


### Installation

To clone the repository:
```bash
git clone https://github.com/jdenholm/TorchTools.git
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


### Run tests
To run the tests:
```bash
pytest tests/
```


### Demonstrations

There are demos for each of the models in the ``demos/`` directory. Note: the demos require a different python environment, which is specified in ``demo-requirements.conda.yaml``. To build it, use:

```bash
cd /path/to/TorchTools
conda env create -f demo-requirements.conda.yaml
conda activate torch-tools-demo
pip install -e .
```
