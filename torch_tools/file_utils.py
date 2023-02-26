"""File searching and path utilities."""
from pathlib import Path
from typing import List, Optional

from zipfile import ZipFile


def ls_zipfile(zip_path: Path) -> List[Path]:
    """List the contents of ``zip_path``.

    Parameters
    ----------
    zip_path : Path
        Path to the zipfile whose contents we want to list.

    Returns
    -------
    List[Path]
        A list of the files in the zipfile at ``zip_path``, sorted by file
        name.

    Raises
    ------
    TypeError
        If ``zip_path`` is not a ``Path``.

    """
    if not isinstance(zip_path, Path):
        raise TypeError(
            f"'{zip_path}' should be Path. Got '{type(zip_path)}'.",
        )

    with ZipFile(zip_path) as zip_archive:
        file_names = zip_archive.namelist()
    return sorted(
        map(lambda x: zip_path / x, file_names),
        key=lambda x: x.name,
    )


def traverse_directory_tree(directory: Path) -> List[Path]:
    """Recursively list all files in ``directory``.

    Parameters
    ----------
    directory : Path
        The directory whose contents should be searched.

    Returns
    -------
    List[Path]
        A list of all of the files in ``directory``, sorted by name.

    Raises
    ------
    TypeError
        If ``directory`` is not a ``Path``.
    FileNotFoundError
        If ``directory`` does not exist.
    RuntimeError
        If ``directory`` is not a directory.

    """
    if not isinstance(directory, Path):
        msg = f"'{directory}' should be a 'Path'. Got '{type(directory)}'."
        raise TypeError(msg)

    if not directory.exists():
        raise FileNotFoundError(directory)

    if not directory.is_dir():
        raise RuntimeError(f"'{directory}' is not a a directory.")

    return sorted(_recursive_search(directory), key=lambda x: x.name)


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
        if item.suffix == ".zip":
            files += ls_zipfile(item)

        if item.is_file():
            files.append(item)

        if item.is_dir():
            _ = _recursive_search(item, files=files)

    return files
