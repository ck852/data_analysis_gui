"""
PatchBatch Electrophysiology Data Analysis Tool
Author: Charles Kissell, Northeastern University
License: MIT (see LICENSE file for details)

Centralized utilities for filename handling used across batch analysis and export
code. Two functions live here:

- `filename_sort_key`: Produces a sort key for electrophysiology filenames so that
  base files (e.g. "260522_001") sort before their decimal sub-versions (e.g.
  "260522_001.1", "260522_001.2", ...) within the same series, and series sort
  hierarchically by (date, experiment, sub-version).

- `clean_filename`: Strips the file extension and any bracketed content (e.g.
  "[1-12]") from a filename, returning a display- and header-friendly name.

Both functions are pure Python with no GUI dependencies, so this module is free
to be imported from anywhere in the codebase, including /core and /services.
"""

import re
from pathlib import Path


def filename_sort_key(name: str) -> tuple:
    """
    Sort key for electrophysiology filenames of the form 'YYMMDD_NNN[.M]'.

    Produces a 3-tuple (date, experiment, sub) so that a base file like
    '260522_001' sorts before its decimal sub-versions ('260522_001.1',
    '260522_001.2', ...). Filenames without a decimal sub-version get sub=0,
    which places them first within their series.

    `name` should be a base name or stem (e.g. "260522_001.1" or
    "260522_001.1[1-12]"), not a full file path containing directory
    components — `re.search` will otherwise match digits inside parent
    directory names.

    Fallback for unconventional names: returns a tuple of every integer
    substring found, or (0,) if none.

    Examples:
        >>> filename_sort_key("260522_001")
        (260522, 1, 0)
        >>> filename_sort_key("260522_001.1")
        (260522, 1, 1)
        >>> filename_sort_key("260522_001.1[1-12]")
        (260522, 1, 1)
        >>> filename_sort_key("260522_002")
        (260522, 2, 0)
    """
    match = re.search(r"(\d+)_(\d+)(?:\.(\d+))?", name)
    if match:
        sub = int(match.group(3)) if match.group(3) else 0
        return (int(match.group(1)), int(match.group(2)), sub)

    numbers = re.findall(r"\d+", name)
    if numbers:
        return tuple(int(n) for n in numbers)

    return (0,)


def clean_filename(file_path: str) -> str:
    """
    Clean a filename for display by stripping the extension and any bracketed
    content (e.g. "[1-12]"). Bracket stripping was originally added to handle
    ABF exports from WinWCP which embed sweep-range metadata in filenames.

    Examples:
        >>> clean_filename("/path/to/260522_001.abf")
        '260522_001'
        >>> clean_filename("260522_001.1[1-12].abf")
        '260522_001.1'
    """
    stem = Path(file_path).stem
    cleaned = re.sub(r"\[.*?\]", "", stem).strip()
    return cleaned