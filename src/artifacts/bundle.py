"""Explicit handoffs from training notebooks to independent analysis notebooks.

Models use JSON constructor specifications plus CPU state dictionaries. Named
prepared data and fitted preprocessing use a trusted-local pickle, necessary
for existing PyG graphs and target transforms. This is not a kernel snapshot:
callers explicitly list the values to save; functions/modules are not saved.
"""
import hashlib
import io
import json
import pickle
from . import pickle_compat as artifact_pickle
import shutil
import tempfile
from pathlib import Path

import torch
from torch import nn

from .checkpoints import build_model, model_spec

SCHEMA_VERSION = 1


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_bundle(directory, values, *, splits, provenance=None):
    """Create an immutable training handoff; never overwrite a completed one.

    ``splits`` explicitly records organoid identifiers and their roles. The
    pickle stores named analysis inputs, not executable notebook definitions.
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Choose a new training run: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".handoff-", dir=directory.parent))
    models, seen = {}, {}

    class Writer(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, nn.Module):
                if id(obj) not in seen:
                    key = f"model_{len(seen):04d}"
                    seen[id(obj)] = key
                    models[key] = model_spec(obj)
                    torch.save({k: v.detach().cpu() for k, v in obj.state_dict().items()}, staging / f"{key}.pt")
                return ("model", seen[id(obj)])
            if torch.is_tensor(obj):
                # CPU tensors also preserve PyG/preprocessor portability without
                # moving or mutating the live training objects.
                buffer = io.BytesIO()
                torch.save(obj.detach().cpu(), buffer)
                return ("tensor", buffer.getvalue())
            return None

    try:
        with (staging / "inputs.pkl").open("wb") as handle:
            Writer(handle, protocol=pickle.HIGHEST_PROTOCOL).dump(dict(values))
        (staging / "splits.json").write_text(json.dumps(splits, indent=2, default=str))
        manifest = dict(schema_version=SCHEMA_VERSION, inputs=sorted(values), models=models,
                        provenance=provenance or {},
                        files={p.name: _sha(p) for p in staging.iterdir()})
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        staging.rename(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return directory


def load_bundle(directory, *, device="cpu"):
    """Load a trusted local handoff, verifying all files before deserialization."""
    directory = Path(directory)
    if not (directory / "manifest.json").exists():
        raise FileNotFoundError(f"No completed training handoff at {directory}. Select the output of its training notebook.")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported training handoff version")
    for name, digest in manifest["files"].items():
        if _sha(directory / name) != digest:
            raise ValueError(f"Training artifact changed: {name}")
    models = {}

    class Reader(artifact_pickle.ArtifactUnpickler):
        def persistent_load(self, identifier):
            kind, key = identifier
            if kind == "tensor":
                return torch.load(io.BytesIO(key), map_location="cpu", weights_only=True)
            if kind != "model":
                raise pickle.UnpicklingError(f"Unknown artifact type {kind}")
            if key not in models:
                model = build_model(manifest["models"][key])
                model.load_state_dict(torch.load(directory / f"{key}.pt", map_location="cpu", weights_only=True))
                models[key] = model.to(device).eval()
            return models[key]

    with (directory / "inputs.pkl").open("rb") as handle:
        values = Reader(handle).load()
    if sorted(values) != manifest["inputs"]:
        raise ValueError("Artifact inputs disagree with manifest")
    return values


def graph_membership(**groups):
    """Record ordered, unique organoid IDs for each named graph partition."""
    result = {}
    for role, graphs in groups.items():
        ids = [str(g.organoid_str) for g in graphs]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate organoid IDs in {role}")
        result[role] = ids
    roles = list(result)
    for i, role in enumerate(roles):
        for other in roles[i + 1:]:
            if set(result[role]) & set(result[other]):
                raise ValueError(f"Overlapping organoids in {role} and {other}")
    return result


def indexed_membership(graphs, splits):
    """Resolve every fold's positional split against the saved ordered cohort."""
    return [dict(fold=split["fold"], **graph_membership(
        train=[graphs[i] for i in split["train_indices"]],
        validation=[graphs[i] for i in split["val_indices"]])) for split in splits]
