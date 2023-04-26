"""Setup for `TorchTools`."""
import os
from setuptools import setup

# pylint: disable=unspecified-encoding
with open(os.path.join(".", "VERSION.txt")) as version_file:
    version = version_file.read().strip()


setup(
    name="torch_tools",
    package_dir={"torch_tools": "torch_tools"},
    author="Jim Denholm",
    version=version,
)
