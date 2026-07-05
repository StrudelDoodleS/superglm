"""Native file dialogs for the local editor app."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def choose_save_path(
    *,
    directory: str | None = None,
    filename: str | None = None,
) -> str | None:
    """Open a native save-file dialog and return the selected path.

    Returns ``None`` when the user cancels. Raises ``RuntimeError`` when no
    usable GUI dialog backend is available in the current Python process.
    """

    if _is_headless_linux():
        raise RuntimeError(
            "Native file dialog unavailable because this Linux process has no DISPLAY "
            "or WAYLAND_DISPLAY. Run from a local GUI session or use Download Edited Model."
        )

    initial_dir = _initial_directory(directory)
    initial_name = _safe_filename(filename)
    errors: list[str] = []
    for chooser in (_choose_with_pyside6, _choose_with_tkinter):
        try:
            chosen = chooser(initial_dir, initial_name)
        except Exception as exc:  # pragma: no cover - depends on optional GUI packages
            errors.append(str(exc))
            continue
        if chosen:
            return _with_default_suffix(chosen)
        return None
    detail = "; ".join(error for error in errors if error)
    message = "Native file dialog unavailable. Install PySide6 or ensure tkinter is available."
    if detail:
        message = f"{message} Details: {detail}"
    raise RuntimeError(message)


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


def _safe_filename(filename: str | None) -> str:
    name = filename or "superglm_edited_model.joblib"
    if Path(name).name != name:
        name = Path(name).name
    if not Path(name).suffix:
        name = f"{name}.joblib"
    return name


def _with_default_suffix(path: str) -> str:
    selected = Path(path).expanduser()
    if not selected.suffix:
        selected = selected.with_suffix(".joblib")
    return str(selected)


def _is_headless_linux() -> bool:
    return (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    )


def _choose_with_pyside6(initial_dir: Path, initial_name: str) -> str | None:
    from PySide6.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    selected, _filter = QFileDialog.getSaveFileName(
        None,
        "Save Edited Model",
        str(initial_dir / initial_name),
        "Joblib model (*.joblib);;All files (*)",
    )
    return selected or None


def _choose_with_tkinter(initial_dir: Path, initial_name: str) -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    try:
        selected = filedialog.asksaveasfilename(
            parent=root,
            title="Save Edited Model",
            initialdir=str(initial_dir),
            initialfile=initial_name,
            defaultextension=".joblib",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*")],
        )
    finally:
        root.destroy()
    return selected or None
