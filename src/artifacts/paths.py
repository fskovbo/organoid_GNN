"""Project paths independent of notebook nesting or launch directory."""
from pathlib import Path
from datetime import datetime
import re


def project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    if start.is_file():
        start = start.parent
    for path in (start, *start.parents):
        if (path / "src/models/gnn.py").is_file():
            return path
    raise FileNotFoundError(f"Cannot locate GraphNN project above {start}")


def training_run_path(root, notebook, tag='', *, timestamp=None):
    """Name a new run without creating it; tags cannot contain path separators."""
    notebook = Path(notebook).stem
    if not re.fullmatch(r'[A-Za-z0-9_-]+', notebook):
        raise ValueError('Invalid training notebook name')
    if not isinstance(tag, str) or (tag and not re.fullmatch(r'[A-Za-z0-9_-]+', tag)):
        raise ValueError('Run tag must be empty or contain only letters, digits, underscores and hyphens')
    timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    if not re.fullmatch(r'\d{8}_\d{6}', timestamp):
        raise ValueError('Timestamp must use YYYYMMDD_HHMMSS')
    return Path(root) / 'training_results' / notebook / (f'{tag}_{timestamp}' if tag else timestamp)


def resolve_training_run(root, notebook, run_name=None):
    """Select a named run, or the sole available run; never silently select latest.

    An explicit absolute or project-relative path also accepts historical runs.
    """
    base = Path(root) / 'training_results' / Path(notebook).stem
    if run_name is not None:
        path = Path(run_name)
        path = path if path.is_absolute() else (Path(root) / path if len(path.parts) > 1 else base / path)
        if not (path / 'settings.json').is_file():
            raise FileNotFoundError(f'No saved training settings at {path}')
        return path
    runs = sorted(p for p in base.glob('*') if (p / 'settings.json').is_file()
                  and ((p / 'models.json').exists() or (p / 'training_complete.json').exists()))
    if len(runs) != 1:
        raise ValueError(f'Set TRAINING_RUN to a run name under {base}. Available runs: '
                         f'{[p.name for p in runs]}')
    return runs[0]
