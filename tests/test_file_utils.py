"""Tests for ``torch_tools.file_utils``."""
from pathlib import Path
from shutil import rmtree

import pytest

from torch_tools.file_utils import traverse_directory_tree

_parent_dir = Path(".test-paths/").resolve()
_base_path = Path(_parent_dir, "Meriadoc/Peregrin/Samwise/Frodo/").resolve()

_paths = sorted(
    [
        _base_path / "Boromir.txt",
        _base_path.parent / "Aragorn.txt",
        _base_path.parent.parent / "Legolas.txt",
        _base_path.parent.parent.parent / "Gimli.txt",
        _base_path.parent.parent.parent / "Gandalf.txt",
    ]
)


@pytest.fixture(scope="module", autouse=True)
def create_directory_tree():
    """Create a directory tree to test with."""
    _base_path.mkdir(parents=True)
    _ = list(map(lambda x: x.touch(), _paths))

    yield

    rmtree(_parent_dir)


def test_traverse_directory_tree_type_checking():
    """Test the ``directory`` argument only accepts ``Path``s."""
    # Should work with paths
    traverse_directory_tree(_parent_dir)

    # Should break with non-path
    with pytest.raises(TypeError):
        traverse_directory_tree(str(_parent_dir))


def test_traverse_directory_tree_directory_not_exists():
    """Test ``directory``s which don't exist are caught."""
    # Should work with existing paths
    traverse_directory_tree(_parent_dir)

    # Should break with paths that don't exist
    with pytest.raises(FileNotFoundError):
        traverse_directory_tree(Path("Minas", "Tirith/"))


def test_traverse_directory_tree_return():
    """Test the list of the returned files."""

    files = sorted(traverse_directory_tree(_parent_dir))

    assert all(map(lambda paths: paths[0] == paths[1], zip(files, _paths)))
