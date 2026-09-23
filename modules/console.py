"""ANSI-цвета для консольного меню (отключаются вне TTY или при NO_COLOR)."""
import os
import sys

_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _wrap(code: str, text: str) -> str:
    if not _ENABLED:
        return text
    return f"\033[{code}m{text}\033[0m"


def c_red(text: str) -> str:
    return _wrap("31", text)


def c_green(text: str) -> str:
    return _wrap("32", text)


def c_yellow(text: str) -> str:
    return _wrap("33", text)


def c_cyan(text: str) -> str:
    return _wrap("36", text)


def c_magenta(text: str) -> str:
    return _wrap("35", text)


def c_bold(text: str) -> str:
    return _wrap("1", text)


def c_dim(text: str) -> str:
    return _wrap("2", text)
