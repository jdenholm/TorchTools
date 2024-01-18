## Version 0.10.0
  - Updated ``torch_tools.weight_init.normal_init`` with options for which attrs get intialised.

## Version 0.9.0
  - Added total variational loss function ``total_image_variation``.

## Version 0.8.0
  - Add ``get_features`` method to ``SimpleConvNet2d`` for extracting encoded features.
  - Add ``get_features`` method to ``ConvNet2d`` for extracting encoded features.
  - Added a demo using a multiple-instance-learning attention model.

## Version 0.7.0
  - Changed ``AutoEncoder2d`` demo to use ovarian histology images.

## Version 0.6.1
  - Added residual blocks as optional block style to all relevant models.
  - Changed the ``UNet`` demo to use a nuclei segmentation data set.

## Version 0.6.0
  - Added synthetic shapes dataset and demo.

## Version 0.5.6
  - Changed the classification demo for ``FCNet`` to the penguin problem.

## Version 0.5.5
  - Added ``py.typed`` package data to toml.

## Version 0.5.4
  - Added missing docs for ``VAE2d``.

## Version 0.5.3
  - Added ``torch_tools.models._blocks_2d.ConvResBlock``.

## Version 0.5.2
  - Removed biases in residual block.

## Version 0.5.1
  - Added ``py.typed`` file.

## Version 0.5.0
  - Added ``torch_tools.VAE2d`` model.

## Version 0.4.3
  - Updated the docstring in ``torch_tools.ConvNet2d``.

## Version 0.4.2
  - Fixed error in the doc-building caused by Torchvision (and possibly PyTorch).

## Version 0.4.1
  - Added a demo for the ``patchify_img_batch`` function.

## Version 0.4.0
Changes:
  - Added mixup augmentation option to ```DataSet``.

## Version 0.3.2
Changes:
  - Fixed typo in ``FCNet`` docstring: ``DenseNetwork`` -> ``FCNet``
  - Minor refactoring of GitHub workflows.
  - Move some of the dev requirements into the pip section.


## Version 0.3.1
Updated the docstring in ``ConvNet2d`` to include the mobilenet options.

## Version 0.3.0
This release introduces:

- Mobilenet v3 as an encoder style for ``ConvNet2d``.
- Fixing a docstring typo in ``ConvNet2d``.

## Version 0.2.2
The only difference in this minor version update is a tiny patch to the README: the instructions for the ``pip install ...`` commands were missing the ``"git+"`` prefixes before the URL. They have now been added.

## Version 0.2.1
The source-code links in docs weren't working. This ha now been fixed.


## Version 0.2.0

### New Features

#### Variable kernel size
For all of the models using 2D convolutional layers (except ``ConvNet2d``, which uses predetermined architectures), the kernel size argument is now optional. Spoiler alert: it has to be an odd, positive, integer.


### Documentation
- Fixed a typo in the docstring for ``torch_tools.models._argument_processing.process_dropout_prob``. The ``prob`` arg should be on ``[0.0, 1.0)``, and not ``(0.0, 1.0]`` as described. This was only a typo in the docstring and not a bug.
- Made ``source`` link available in the docs.

### Python package
- The repo has moved from the old ``setup.py`` to use a ``pyproject.toml`` — hopefully correctly. The version imports and python dependencies have been updated accordingly.
- The demos now use the ``requirements-dev.conda.yaml``.
- You can now install the package with pip from Github.
