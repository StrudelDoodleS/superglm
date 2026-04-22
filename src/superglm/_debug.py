from __future__ import annotations

import os

DEBUG = int(os.environ.get("SUPERGLM_DEBUG", "0"))


def set_debug_level(level: int) -> None:
    global DEBUG
    DEBUG = int(level)


def get_debug_level() -> int:
    return int(DEBUG)


def debug_enabled(level: int) -> bool:
    return get_debug_level() >= level
