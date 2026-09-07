"""Deliberate validation messages that are safe to show in the editor."""


class EditorClientError(Exception):
    """Only construct this with an intentional message, never backend exception text."""

    def __init__(self, public_message: str):
        super().__init__(public_message)
        self.public_message = public_message


class EditorValueError(EditorClientError, ValueError):
    """Invalid editor input; remains a ValueError for Python callers."""


class EditorKeyError(EditorClientError, KeyError):
    """Unknown editor selection; remains a KeyError for Python callers."""


class EditorTypeError(EditorClientError, TypeError):
    """Unsupported editor operation; remains a TypeError for Python callers."""


class EditorIndexError(EditorClientError, IndexError):
    """Out-of-range editor selection; remains an IndexError for Python callers."""
