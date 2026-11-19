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


def no_quote_message(error, *, counted: str | None = None) -> str:
    """The page's message for rows ``predict`` gave no mean (``NoPredictedMeanError``).

    ``counted`` words their count, as a fold's held-out rows.
    """
    from superglm.model.base import _shown_positions

    counted = counted or f"{error.positions.size} of {error.n_rows} rows"
    return (
        f"{counted} have no quote under this model (row positions "
        f"{_shown_positions(error.positions)}): the {error.link} has no mean at their linear "
        "predictor. These scores are not shown while those rows are in the data."
    )
