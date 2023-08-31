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
