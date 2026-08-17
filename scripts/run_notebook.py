#!/usr/bin/env python3
"""Executes a notebook top-to-bottom headlessly, coalesces noisy stream output (e.g.
tqdm progress bars writing many small chunks) into one block per run so the saved
file stays readable, and fails loudly on any cell error.

Run from the project root, using your venv's python (path varies per machine):
    <path-to-venv>\Scripts\python.exe scripts/run_notebook.py notebooks/<name>.ipynb
"""

import sys

import nbformat
from nbclient import NotebookClient

KERNEL_NAME = "plant_disease"


def coalesce_streams(nb):
    """Merges consecutive same-name stream outputs (stdout/stdout, stderr/stderr)
    within a cell into one -- without this, tqdm's frequent small writes turn into
    dozens of separate output objects per cell instead of one updating block."""
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        merged = []
        for output in cell.get("outputs", []):
            if (
                output.get("output_type") == "stream"
                and merged
                and merged[-1].get("output_type") == "stream"
                and merged[-1].get("name") == output.get("name")
            ):
                merged[-1]["text"] += output["text"]
            else:
                merged.append(output)
        cell["outputs"] = merged
    return nb


def run(path: str, timeout: int = 600) -> int:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, kernel_name=KERNEL_NAME, timeout=timeout, resources={"metadata": {"path": "."}})
    client.execute()
    coalesce_streams(nb)
    nbformat.write(nb, path)

    errors = [
        (i, o) for i, c in enumerate(nb["cells"]) for o in c.get("outputs", []) if o.get("output_type") == "error"
    ]
    if errors:
        print(f"ERRORS: {len(errors)}")
        for i, o in errors:
            print(f"cell {i}: {o.get('ename')}: {o.get('evalue')}")
        return 1

    print(f"{path}: executed cleanly, 0 errors.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_notebook.py notebooks/<name>.ipynb (run from the project root)")
        sys.exit(2)
    sys.exit(run(sys.argv[1]))
