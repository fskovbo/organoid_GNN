"""Fingerprint executable notebook inputs independently of displayed outputs."""
import hashlib
import json
from pathlib import Path

from .paths import project_root


def workflow_fingerprint(notebook, *, source_roots=("src",)):
    """Hash ordered code cells and the maintained Python sources they can use.

    Output images, execution counts, notebook IDs, and markdown are excluded.
    The default deliberately invalidates computation caches on any src change;
    callers can supply narrower source roots when dependencies are bounded.
    Historical completed outputs remain readable without this computation check.
    """
    notebook = Path(notebook)
    root = project_root(notebook)
    cells = json.loads(notebook.read_text())["cells"]
    sources = {str(notebook.relative_to(root)): ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]}
    for relative in source_roots:
        path = root / relative
        paths = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for source in paths:
            sources[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
