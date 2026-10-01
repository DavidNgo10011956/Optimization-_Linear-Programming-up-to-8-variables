"""
lp_utils.py
===========
Small layout and input helpers shared by the terminal programs in the toolkit
(the LP solver and the integral solver use the same guide format).

Named lp_utils (not utils) on purpose: a PyPI package called 'utils' is
commonly installed, and a plain 'import utils' can pick that one up instead.

    WIDTH            -- width of the banner / horizontal rules
    hr(char)         -- a horizontal rule, e.g. hr() or hr("─")
    format_table(...) -- a fixed-width text table with word-wrapped cells
    parse_name_list  -- "x, y, z" -> ["x", "y", "z"] with validation
"""

from __future__ import annotations

import re
import textwrap
from typing import Dict, List, Sequence

WIDTH = 92

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def hr(char: str = "=", width: int = WIDTH) -> str:
    """Return a horizontal rule made of `char`, `width` characters long."""
    char = char or "="
    return (char * width)[:width]


def format_table(headers: Sequence[str], rows: Sequence[Dict[str, str]],
                 keys: Sequence[str], widths: Sequence[int], indent: int = 2) -> List[str]:
    """Render a fixed-width text table and return it as a list of lines.

    headers -- column titles shown in the header row
    rows    -- one dict per row
    keys    -- which dict key feeds each column (lets a header differ from its key)
    widths  -- width of each column in characters; long cells wrap onto extra lines
    """
    if not (len(headers) == len(keys) == len(widths)):
        raise ValueError("headers, keys and widths must have the same length.")

    pad = " " * indent
    sep = "  "

    def render(cells: Sequence[str]) -> List[str]:
        wrapped = [textwrap.wrap(str(cell), w) or [""] for cell, w in zip(cells, widths)]
        height = max(len(w) for w in wrapped)
        out = []
        for i in range(height):
            parts = [(col[i] if i < len(col) else "").ljust(w) for col, w in zip(wrapped, widths)]
            out.append((pad + sep.join(parts)).rstrip())
        return out

    lines = render(headers)
    lines.append(pad + sep.join("-" * w for w in widths))
    for row in rows:
        lines.extend(render([row.get(k, "") for k in keys]))
    return lines


def parse_name_list(text: str) -> List[str]:
    """Split 'x, y, z' (commas and/or spaces) into validated, unique names.

    Names must start with a letter or underscore and contain only letters,
    digits and underscores -- the same rule the constraint parser uses, so
    every accepted name can actually be typed in a constraint later.
    """
    names = [n for n in re.split(r"[,\s]+", text.strip()) if n]
    if not names:
        raise ValueError("Please enter at least one variable name.")
    for name in names:
        if not _NAME_RE.match(name):
            raise ValueError(
                f"'{name}' is not a valid name: start with a letter and use only letters, "
                "digits or _ (e.g. x1, bread, tables)."
            )
    seen = set()
    dupes = [n for n in names if n in seen or seen.add(n)]
    if dupes:
        raise ValueError(f"Variable names must be unique (repeated: {', '.join(sorted(set(dupes)))}).")
    return names
