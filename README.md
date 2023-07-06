# TorchTools
<p align="left">
  <a href="https://github.com/jdenholm/TorchTools/releases/"><img alt="" src="https://img.shields.io/github/v/release/jdenholm/TorchTools" /></a>
  <a href="https://github.com/jdenholm/TorchTools/blob/dev/LICENSE.md"><img alt="" src="https://img.shields.io/github/license/jdenholm/TorchTools" /></a>
  <a href=""><img alt="" src="" /></a>
</p>

Handy PyTorch models and utilities which should save you time—and heartache.

## Please **read the** [**documentation**](https://jdenholm.github.io/TorchTools/).





## Description
This Python library contains a collection of neural networks, implemented in PyTorch, and a generic ``DataSet`` class with useful features to help you get started with your deep-learning problem more quickly. Hopefully, the user will spend less time engineering and testing—because researchers *always* test —their code. There is also the hope that you will write and duplicate less code by being able to easily set and modify these models' architectures at instantiation.


## Who is this repository for?
This repository is for anyone intested in using the tools—deep learning or otherwise—it contains. Please cite the repository if you use it in any work you publish. Please also contact me, by creating an issue on GitHub, with details of the publication so I can maintain a list of publications in this README.

## Authors and Acknowledgement
This project was created by [J. Denholm](https://github.com/jdenholm).

J.D. kindly thanks:
- [J. Atkinson](https://github.com/jatkinson1000)
- [A. Haglund](https://github.com/ah3918)
- [E. Pascal](https://github.com/elena-pascal)

for their help and time in reviewing demos for this repo.

## Contributions
If you would like to contribute to this repository, or request something—such as a particular model—be added to it, simply create an issue on [GitHub](https://github.com/jdenholm/TorchTools) and I'll be glad to discuss collaborating or adding the model.

## License
Distributed under the MIT License. See LICENSE.md for more information.


## Getting Started


### Installation
To install the most up-to-date version, use
```bash
pip install git+https://github.com/jdenholm/TorchTools.git
```
To install a specific version, say ``v0.1.0``, use
```bash
pip install git+https://github.com/jdenholm/TorchTools.git@v0.1.0
```


### Run tests
To run the tests (with the dev env):
```bash
pytest tests/
```


### Demonstrations

There are demos for each of the models in the ``demos/`` directory. Note: the demos require a different python environment, which is specified in [``requirements-dev.conda.yaml``](requirements-dev.conda.yaml). To build it, use:

```bash
cd /path/to/TorchTools/
conda env create -f requirements-dev.conda.yaml
```