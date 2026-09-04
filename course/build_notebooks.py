"""Build runnable .ipynb notebooks from the day markdown files (jupytext "light" format).

Layout:  course/dayNN-<slug>/dayNN-<slug>.md  ->  dayNN-<slug>/dayNN-<slug>.ipynb

Rules:
- Markdown lines              -> markdown cells (the theory)
- Blocks between "# %%" lines -> code cells
- "# %% [markdown]"           -> switch back to markdown mode

Usage:
  python build_notebooks.py            build every notebook next to its .md
  python build_notebooks.py --check    only verify that every code cell compiles
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def to_notebook(text: str) -> dict:
    cells = []
    buf: list[str] = []   # markdown buffer
    code: list[str] | None = None

    def flush_md() -> None:
        nonlocal buf
        if buf:
            while buf and not buf[0].strip():
                buf.pop(0)
            while buf and not buf[-1].strip():
                buf.pop()
            if buf:
                cells.append({"cell_type": "markdown", "metadata": {}, "source": ["\n".join(buf)]})
            buf = []

    def flush_code() -> None:
        nonlocal code
        if code is not None:
            while code and not code[0].strip():
                code.pop(0)
            while code and not code[-1].strip():
                code.pop()
            if code:
                cells.append({"cell_type": "code", "execution_count": None,
                              "metadata": {}, "outputs": [],
                              "source": ["\n".join(code)]})
            code = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "# %% [markdown]":
            flush_md()
            flush_code()
            code = None
        elif stripped == "# %%" or stripped.startswith("# %% "):
            flush_md()
            flush_code()
            code = []
        elif code is not None:
            code.append(line)
        else:
            buf.append(line)
    flush_md()
    flush_code()

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def strip_magics(src: str) -> str:
    """Make a cell's code py_compile-clean.

    - Cells starting with a cell magic (%%writefile ...) contain the written
      file's body (Python, YAML, ...) -> drop the whole cell.
    - Shell magics (!cmd) are single lines inside otherwise-Python cells -> drop
      only those lines.
    """
    lines = src.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        return ""
    return "\n".join(line for line in lines if not line.lstrip().startswith("!"))


def day_files():
    return sorted(p for p in HERE.glob("day*/*.md") if p.stem.startswith("day"))


def build_all() -> None:
    for md in day_files():
        nb = to_notebook(md.read_text(encoding="utf-8"))
        (md.parent / (md.stem + ".ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"built {md.parent.name}/{md.stem}.ipynb  ({len(nb['cells'])} cells)")


def validate_all() -> int:
    errors = 0
    cells = 0
    for md in day_files():
        nb = to_notebook(md.read_text(encoding="utf-8"))
        for j, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            cells += 1
            src = strip_magics(cell["source"][0])
            if not src.strip():
                continue
            try:
                compile(src, f"{md.stem} cell{j}", "exec")
            except SyntaxError as e:
                errors += 1
                print(f"SYNTAX {md.parent.name}/{md.stem} cell {j} line {e.lineno}: {e.msg}")
                line = src.splitlines()[e.lineno - 1] if e.lineno else ""
                print(f"   {line[:90]}")
    print(f"{cells} code cells checked, {errors} error(s)")
    return errors


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(validate_all())
    build_all()
