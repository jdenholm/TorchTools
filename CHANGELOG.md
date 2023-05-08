## Version 0.2.0


### Documentation
- Fixed a typo in the docstring for ``torch_tools.models._argument_processing.process_dropout_prob``. The ``prob`` arg should be on ``[0.0, 1.0)``, and not ``(0.0, 1.0]`` as described. This was only a typo in the docstring and not a bug.