"""Terminal UI helpers for the interactive wizard."""

from __future__ import annotations

import sys
from pathlib import Path


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(text: str) -> str:
    return _c("1", text)


def dim(text: str) -> str:
    return _c("2", text)


def green(text: str) -> str:
    return _c("32", text)


def yellow(text: str) -> str:
    return _c("33", text)


def cyan(text: str) -> str:
    return _c("36", text)


def red(text: str) -> str:
    return _c("31", text)


def banner(title: str, subtitle: str = "") -> None:
    line = "═" * 60
    print()
    print(bold(line))
    print(bold(f"  {title}"))
    if subtitle:
        print(dim(f"  {subtitle}"))
    print(bold(line))
    print()


def step_header(number: int, total: int, title: str) -> None:
    print()
    print(cyan(f"━━ Step {number}/{total}: {title} ━━"))
    print()


def info(text: str) -> None:
    for line in text.strip().splitlines():
        print(f"  {dim('→')} {line}")
    print()


def success(text: str) -> None:
    print(green(f"  ✓ {text}"))


def warn(text: str) -> None:
    print(yellow(f"  ! {text}"))


def error(text: str) -> None:
    print(red(f"  ✗ {text}"))


def pause(message: str = "Press Enter to continue…") -> None:
    try:
        input(dim(f"\n  {message} "))
    except (EOFError, KeyboardInterrupt):
        print()
        raise


def ask_yes_no(prompt: str, *, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            raw = input(f"  {prompt} {suffix} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            raise
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        warn("Please answer y or n.")


def ask_text(prompt: str, *, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        raw = input(f"  {prompt}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise
    return raw or default


def ask_path(prompt: str, *, must_exist: bool = True, default: str = "") -> Path:
    while True:
        raw = ask_text(prompt, default=default)
        if not raw:
            warn("Path is required.")
            continue
        path = Path(raw).expanduser()
        if must_exist and not path.exists():
            error(f"Not found: {path}")
            continue
        return path.resolve()


def ask_choice(prompt: str, options: list[tuple[str, str]], *, default: int = 0) -> str:
    """Return the key of the selected option."""
    print(f"  {prompt}")
    for i, (key, label) in enumerate(options):
        mark = green("●") if i == default else " "
        print(f"    {mark} {i + 1}. {label} {dim(f'({key})')}")
    while True:
        try:
            raw = input(f"  Choose [1-{len(options)}, default {default + 1}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise
        if not raw:
            return options[default][0]
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        # allow typing key directly
        for key, _ in options:
            if raw.lower() == key.lower():
                return key
        warn(f"Enter 1–{len(options)} or a valid option key.")
