"""Exercise the actual notebook-owned study procedures on small test fixtures.

Top-level imports and tagged definition cells are executed; settings, data
loading, training invocations, and saved outputs are never executed.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def workflow(path, name):
    namespace = {"PROJECT_ROOT": ROOT, "ROOT": ROOT}
    notebook = json.loads((ROOT / "experiments" / path).read_text())
    for index, cell in enumerate(notebook["cells"]):
        if "workflow-definitions" in cell.get("metadata", {}).get("tags", []):
            exec(compile("".join(cell["source"]), f"{path}:cell{index}", "exec"), namespace)
        elif cell['cell_type'] == 'code':
            imports = [node for node in ast.parse(''.join(cell['source'])).body
                       if isinstance(node, (ast.Import, ast.ImportFrom))]
            exec(compile(ast.Module(body=imports, type_ignores=[]), f"{path}:imports{index}", "exec"), namespace)
    return namespace[name]
