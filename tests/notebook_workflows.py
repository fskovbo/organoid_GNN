"""Exercise the actual notebook-owned study procedures on small test fixtures.

Only tagged definition cells are executed; settings, data loading, training
invocations, and saved outputs are never executed by this loader.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def workflow(path, name):
    namespace = {"PROJECT_ROOT": ROOT, "ROOT": ROOT}
    notebook = json.loads((ROOT / "experiments" / path).read_text())
    for index, cell in enumerate(notebook["cells"]):
        if "workflow-definitions" in cell.get("metadata", {}).get("tags", []):
            exec(compile("".join(cell["source"]), f"{path}:cell{index}", "exec"), namespace)
    return namespace[name]
