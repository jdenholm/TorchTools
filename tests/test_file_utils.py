"""Tests for ``torch_tools.file_utils``."""
from pathlib import Path
from shutil import rmtree, make_archive
from zipfile import BadZipFile

import pytest

from torch_tools.file_utils import traverse_directory_tree, ls_zipfile

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


@pytest.fixture(scope="module", autouse=True)
def create_zipfile():
    """Create a zipfile to test with."""
    path = Path(".good_zip/")
    path.mkdir()
    for idx in range(10):
        (path / f"{idx}.txt").touch()
    make_archive(str(path), "zip", path)
    rmtree(path)

    yield

    path.with_suffix(".zip").unlink()


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


def test_traverse_directory_tree_with_non_dir():
    """Test ``directory`` argument only accepts directories."""
    # Should work with existing paths
    traverse_directory_tree(_parent_dir)

    # Should break with existing paths which are not directories
    with pytest.raises(RuntimeError):
        traverse_directory_tree(Path(__file__))


def test_traverse_directory_tree_return():
    """Test the list of the returned files."""

    files = sorted(traverse_directory_tree(_parent_dir))

    assert all(map(lambda paths: paths[0] == paths[1], zip(files, _paths)))


def test_ls_zipfile_argument_type():
    """Test the ``zip_path`` argument only accepts ``Path``s.

    Notes
    -----
    See the pytest fixture ``create_zipfile``.

    """
    # Should work with Path
    ls_zipfile(Path(".good_zip.zip"))

    # Should break with non-path
    with pytest.raises(TypeError):
        ls_zipfile(".good_zip.zip")


def test_ls_zipfile_with_non_existing_file():
    """Test with a file that doesn't exist."""
    with pytest.raises(FileNotFoundError):
        ls_zipfile(Path("Eowyn.zip"))


def test_ls_zipfile_with_directory():
    """Test with a directory and not a zip file."""
    with pytest.raises(IsADirectoryError):
        ls_zipfile(Path(__file__).parent)


def test_ls_zipfile_with_non_zipfile():
    """Test with a file which is not a zip file."""
    with pytest.raises(BadZipFile):
        ls_zipfile(Path(__file__))


def test_ls_zipfile_return_types():
    """Test the return is of type ``List[Path]``."""
    out_list = ls_zipfile(Path(".good_zip.zip"))
    assert isinstance(out_list, list), "Should return a list"

    msg = "Should only contain Paths"
    assert all(map(lambda x: isinstance(x, Path), out_list)), msg


def test_ls_zipfile_return_values():
    """Test the returned values are correct."""
    expected = [Path(".good_zip.zip") / f"{idx}.txt" for idx in range(10)]

    returned = ls_zipfile(Path(".good_zip.zip"))

    for exp, ret in zip(expected, returned):

        assert exp == ret
