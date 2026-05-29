"""
Generate an executed Jupyter notebook for every script in the ladder.

Each notebook gets:
  1. a markdown cell  — title, tagline, the description + key formula + theory
                        (reused from build_site.py so it stays in sync)
  2. a code cell      — the script's imports and definitions
  3. a code cell      — the script's __main__ body (dedented), which RUNS and
                        shows its output inline

The notebooks are then EXECUTED so their outputs are embedded — open them in
Jupyter / VS Code, or view them on GitHub (which renders outputs).

Run with:  .venv/bin/python build_notebooks.py
"""

import pathlib
import textwrap

import nbformat as nbf
from nbclient import NotebookClient

import build_site as bs   # reuse STEPS, THEORY, esc, read_source, eq_theory_html

ROOT = pathlib.Path(__file__).parent
KERNEL = "llmvenv"


def split_source(src):
    """Return (definitions, main-body) with the module docstring removed and the
    `if __name__ == '__main__':` block dedented so it runs at top level."""
    s = src.lstrip()
    if s[:3] in ('"""', "'''"):
        q = s[:3]
        s = s[s.index(q, 3) + 3:]
    s = s.lstrip("\n")
    marker = '\nif __name__ == "__main__":'
    if marker in s:
        pre, post = s.split(marker, 1)
        run = textwrap.dedent(post.lstrip("\n")).strip("\n")
        return pre.strip("\n"), run
    return s.strip("\n"), ""


def markdown(s):
    parts = [f"# {s['n']} · {bs.esc(s['title'])}", f"**{s['tag']}**"]
    parts += s["body"]
    eqth = bs.eq_theory_html(s)
    if eqth:
        parts.append(eqth)
    parts.append(f'<p style="color:#888"><em>Source: <code>{s["file"]}</code> · '
                 f'run the cells below to reproduce the output.</em></p>')
    return "\n\n".join(parts)


def build_notebook(s):
    pre, run = split_source(bs.read_source(s["file"]))
    cells = [nbf.v4.new_markdown_cell(markdown(s))]
    if pre:
        cells.append(nbf.v4.new_code_cell(pre))
    if run:
        cells.append(nbf.v4.new_code_cell(run))
    nb = nbf.v4.new_notebook(cells=cells)
    return nb


def main():
    for s in bs.STEPS:
        nb = build_notebook(s)
        name = s["file"].replace(".py", ".ipynb")
        try:
            NotebookClient(
                nb, timeout=900, kernel_name=KERNEL, allow_errors=True,
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
            status = "executed"
        except Exception as e:                       # pragma: no cover
            status = f"exec error: {e}"
        # Portable kernelspec so the file opens anywhere, not just our venv.
        nb.metadata["kernelspec"] = {
            "name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        nbf.write(nb, str(ROOT / name))
        n_out = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
        print(f"{name:28s} {status:10s} ({n_out} output blocks)")


if __name__ == "__main__":
    main()
