"""Load image/label pair lists from CSV or JSON."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

_IMAGE = ("image", "image_url")
_LABEL = ("label", "label_path")
_RESERVED = set(_IMAGE + _LABEL)
_SPLITS = frozenset({"trn", "tst"})


def parse_split(value, *, src: str | Path | None = None) -> str:
    if value is None or pd.isna(value):
        loc = f"{src}: " if src else ""
        raise ValueError(f"{loc}split is empty")
    s = str(value).strip().lower()
    if s not in _SPLITS:
        loc = f"{src}: " if src else ""
        raise ValueError(f"{loc}invalid split {value!r}; expected trn or tst")
    return s


def resolve_split(metadata: dict | None, *, src: str | Path | None = None) -> str:
    meta = metadata or {}
    if "split" not in meta:
        loc = f"{src}: " if src else ""
        raise ValueError(f"{loc}split is required (trn or tst)")
    return parse_split(meta["split"], src=src)


def _first(rec: dict, names: tuple[str, ...]):
    for n in names:
        v = rec.get(n)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return v
    return None


def load_pairs(*paths: str | Path, trn_only: bool = False) -> list[dict]:
    """Return ``[{image, label, metadata}, ...]``.

    CSV requires ``split`` (trn/tst) plus ``image``/``image_url`` and
    ``label``/``label_path``. Extra columns become metadata. JSON is a list
    of the same dicts with ``split`` in metadata.
    """
    pairs: list[dict] = []
    for path in paths:
        pairs.extend(_load_one(Path(path), trn_only=trn_only))
    return pairs


def _load_one(path: Path, trn_only: bool) -> list[dict]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"{path}: JSON must be a list of pair dicts")
        out = []
        for item in data:
            meta = dict(item.get("metadata") or {})
            meta["split"] = resolve_split(meta, src=path)
            if trn_only and meta["split"] != "trn":
                continue
            out.append(
                {"image": item["image"], "label": item.get("label"), "metadata": meta}
            )
        return out

    if path.suffix.lower() != ".csv":
        raise ValueError(f"{path}: use .json or .csv")

    df = pd.read_csv(path)
    if "split" not in df.columns:
        raise ValueError(f"{path}: missing split column (trn or tst)")
    rows = []
    for rec in df.to_dict(orient="records"):
        image = _first(rec, _IMAGE)
        if image is None:
            raise ValueError(f"{path}: missing image/image_url")
        meta = {k: v for k, v in rec.items() if k not in _RESERVED and pd.notna(v)}
        meta["split"] = parse_split(rec.get("split"), src=path)
        if trn_only and meta["split"] != "trn":
            continue
        rows.append(
            {"image": image, "label": _first(rec, _LABEL), "metadata": meta}
        )
    return rows


def parse_class_ids(raw: str | None, *, required: bool = False) -> dict | None:
    if raw is None:
        if required:
            raise ValueError("--class_ids is required")
        return None
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--class_ids must be a dict literal")
    return parsed


def parse_attr_values(raw: list[str] | None) -> list | None:
    if raw is None:
        return None
    out = []
    for v in raw:
        try:
            out.append(int(v))
        except ValueError:
            out.append(v)
    return out
