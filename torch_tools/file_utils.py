"""File searching and path utilities."""
from typing import List, Optional

from pathlib import Path


def traverse_directory_tree(directory: Path) -> List[Path]:
    """Recursively list all files in ``directory``.

    Parameters
    ----------
    directory : Path
        The director whose contents to be searched.

    Returns
    -------
    List[Path]
        A list of all of the files in ``directory``.

    Raises
    ------
    TypeError
        If ``directory`` is not a ``Path``.
    FileNotFoundError
        If ``directory`` does not exist.

    """
    if not isinstance(directory, Path):
        msg = f"'{directory}' should be a 'Path'. Got '{type(directory)}'."
        raise TypeError(msg)
    if not directory.exists():
        raise FileNotFoundError(directory)
    return _recursive_search(directory)


def _recursive_search(
    directory: Path,
    files: Optional[List[Path]] = None,
) -> List[Path]:
    """Recursively descend through ``directory`` and list the files.

    Parameters
    ----------
    directory : Path
        The directory to recursively search.
    files : List[Path]
        A list of files to append new ``Path``s to.

    Returns
    -------
    files : List[Path]
        A list of file paths.

    """
    files = [] if files is None else files
    for item in directory.iterdir():

        if item.is_file():
            files.append(item)

        if item.is_dir():
            _ = _recursive_search(item, files=files)

    return files
