"""Native OS helpers for the local editor app."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def open_directory_path(path: str | Path | None = None) -> Path:
    """Open a directory in the user's OS file manager and return the resolved path."""

    target = _initial_directory(None if path is None else str(path))
    if sys.platform.startswith("win"):
        os.startfile(str(target))  # type: ignore[attr-defined]
        return target
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
        return target

    commands = _open_directory_commands(target)
    for command in commands:
        executable = command[0]
        if shutil.which(executable):
            _launch_directory_command(command)
            return target
    raise RuntimeError(
        "Could not find a desktop file manager opener. Install xdg-open/gio/kde-open "
        "or open the directory manually."
    )


def _open_directory_commands(target: Path) -> tuple[tuple[str, ...], ...]:
    target_arg = str(target)
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    if "kde" in desktop:
        return (
            ("dolphin", "--new-window", target_arg),
            ("kioclient5", "exec", target_arg),
            ("kioclient", "exec", target_arg),
            ("kde-open5", target_arg),
            ("kde-open", target_arg),
            ("xdg-open", target_arg),
            ("gio", "open", target_arg),
        )
    return (
        ("xdg-open", target_arg),
        ("gio", "open", target_arg),
        ("dolphin", target_arg),
        ("kde-open5", target_arg),
        ("kde-open", target_arg),
    )


def _launch_directory_command(command: tuple[str, ...]) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.1)
    if process.poll() is None:
        return
    _stdout, stderr = process.communicate(timeout=0.1)
    if process.poll() == 0:
        return
    detail = stderr.strip() or f"{command[0]} exited with status {process.poll()}"
    raise RuntimeError(detail)


def _initial_directory(directory: str | None) -> Path:
    candidate = Path(directory or Path.cwd()).expanduser()
    if candidate.exists() and not candidate.is_dir():
        candidate = candidate.parent
    if not candidate.exists():
        candidate = Path.cwd()
    return candidate.resolve()
