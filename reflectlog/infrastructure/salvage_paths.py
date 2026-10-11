"""Filesystem ancestry checks shared by offline-salvage adapters."""

import os


def is_directory_within(directory: str, protected: str) -> bool:
    """Return True if the resolved directory is protected or lies below it."""
    current = os.path.realpath(directory)
    root = os.path.realpath(protected)
    while True:
        try:
            if os.path.samefile(current, root):
                return True
        except OSError:
            # Missing ancestors cannot match; continue checking existing parents.
            pass
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent
