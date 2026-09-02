"""Pre-tiling split planner: one ``split_plan.json`` of map-coordinate val rects.

Scans image/label pairs across sensors, then the tiler consumes the plan
read-only. Assigned images stay frozen on rerun; only the shortfall is filled.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds as window_from_bounds

from geotiff_tiler.lib.geo import check_label_type
from geotiff_tiler.lib.io import load_vector_mask, require_class_ids, resolve_attr_field

logger = logging.getLogger(__name__)

PLAN_VERSION = 1
GRID_VERSION = 1


def grid_spec(patch_size: tuple[int, int], stride: int) -> dict:
    h, w = patch_size
    return {"version": GRID_VERSION, "patch": [int(h), int(w)], "stride": int(stride)}


def assert_plan_params(
    plan: dict,
    patch_size: tuple[int, int],
    stride: int,
    label_threshold: float,
    min_valid_frac: float,
) -> None:
    p = plan.get("params") or {}
    sp = p.get("patch_size")
    if isinstance(sp, int):
        stored_patch = [sp, sp]
    elif sp is not None:
        stored_patch = [int(sp[0]), int(sp[1])]
    else:
        stored_patch = None
    got = {
        "patch_size": stored_patch,
        "stride": p.get("stride"),
        "label_threshold": p.get("label_threshold"),
        "min_valid_frac": p.get("min_valid_frac"),
    }
    want = {
        "patch_size": [int(patch_size[0]), int(patch_size[1])],
        "stride": int(stride),
        "label_threshold": float(label_threshold),
        "min_valid_frac": float(min_valid_frac),
    }
    same = got["patch_size"] == want["patch_size"] and got["stride"] == want["stride"]
    for k in ("label_threshold", "min_valid_frac"):
        a, b = got[k], want[k]
        same = same and a is not None and math.isclose(float(a), b, abs_tol=1e-9)
    if same:
        return
    raise ValueError(
        f"split plan params {got!r} != tiler {want!r}. "
        "Re-run --split_planner with the same patch/stride/thresholds."
    )


def window_origins(length: int, patch: int, stride: int) -> list[int]:
    last = length - patch
    origs = list(range(0, last + 1, stride))
    if origs[-1] != last:
        origs.append(last)
    return origs


def require_split(metadata: dict, image) -> str:
    split = str(metadata.get("split") or "").strip().lower()
    if split not in ("trn", "tst"):
        raise ValueError(
            f"{image}: metadata['split'] must be trn or tst, "
            f"got {metadata.get('split')!r}"
        )
    return split


def assign_pair_ids(pairs: list) -> list:
    """``{row}_{image_name}`` on each pair. Existing ids are kept."""
    for i, item in enumerate(pairs):
        meta = item.setdefault("metadata", {})
        if meta.get("pair_id"):
            continue
        name = meta.get("id") or Path(str(item.get("image") or "")).stem or "image"
        meta["pair_id"] = f"{i}_{name}"
    return pairs


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_plan(plan_file: str | Path) -> dict:
    """Load a plan, or return an empty one if the file does not exist."""
    plan_file = Path(plan_file)
    if plan_file.exists():
        with open(plan_file) as f:
            return json.load(f)
    return {
        "version": PLAN_VERSION,
        "created": _now(),
        "updated": None,
        "params": {},
        "target_distribution": {},
        "global": {"total_class_counts": {}, "val_class_counts": {}},
        "sensors": {},
        "images": {},
    }


def save_plan(plan: dict, plan_file: str | Path) -> None:
    plan["updated"] = _now()
    tmp = Path(str(plan_file) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(plan, f, indent=2)
    tmp.replace(plan_file)


def analyze_image(
    image_path: str,
    label_path: str,
    class_ids: dict[str, int],
    attr_fields: str | Sequence[str] | None,
    attr_values: Sequence[int] | None,
    patch_size: int,
    stride: int,
    cell_strides: int = 4,
    coarse_factor: int = 2,
    label_threshold: float = 0.01,
    min_valid_frac: float = 0.5,
) -> dict | None:
    """Coarse-grid analysis of one pair over the image ∩ label intersection.

    Cell rectangles are stored in the image CRS so they survive clip/reproject.
    """
    with rasterio.open(image_path) as src:
        img_crs = src.crs
        img_bounds = src.bounds
        img_transform = src.transform
        native_res = abs(img_transform.a)

    label_bounds, geoms = _read_label(
        label_path, img_crs, attr_fields, attr_values or []
    )
    if geoms is not None and len(geoms) == 0:
        logger.warning("%s: no label features after attribute filter", image_path)
        return None

    inter = (
        max(img_bounds.left, label_bounds[0]),
        max(img_bounds.bottom, label_bounds[1]),
        min(img_bounds.right, label_bounds[2]),
        min(img_bounds.top, label_bounds[3]),
    )
    if inter[0] >= inter[2] or inter[1] >= inter[3]:
        logger.warning("%s: no overlap with label", image_path)
        return None
    win = window_from_bounds(*inter, transform=img_transform)
    col0, row0 = int(math.floor(win.col_off)), int(math.floor(win.row_off))
    width = int(math.ceil(win.width - 1e-6))
    height = int(math.ceil(win.height - 1e-6))
    if width < patch_size or height < patch_size:
        logger.warning("%s: intersection smaller than patch size", image_path)
        return None
    x0, y1 = img_transform * (col0, row0)
    x1, y0 = img_transform * (col0 + width, row0 + height)

    cw = math.ceil(width / coarse_factor)
    ch = math.ceil(height / coarse_factor)
    coarse_tr = transform_from_bounds(x0, y0, x1, y1, cw, ch)
    if geoms is not None:
        coarse = rasterio.features.rasterize(
            geoms,
            out_shape=(ch, cw),
            transform=coarse_tr,
            fill=0,
            dtype="uint8",
        )
    else:
        with rasterio.open(label_path) as lsrc:
            lwin = window_from_bounds(x0, y0, x1, y1, transform=lsrc.transform)
            coarse = lsrc.read(
                1, window=lwin, out_shape=(ch, cw), boundless=True, fill_value=0
            ).astype("uint8")

    n_classes = max(class_ids.values()) + 1
    total_counts = np.bincount(coarse.ravel(), minlength=n_classes)[:n_classes]

    cell_px = cell_strides * stride
    ncx = math.ceil(width / cell_px)
    ncy = math.ceil(height / cell_px)
    cells = []
    for cy in range(ncy):
        for cx in range(ncx):
            px0, py0 = cx * cell_px, cy * cell_px
            px1 = min(px0 + cell_px, width)
            py1 = min(py0 + cell_px, height)
            block = coarse[
                py0 // coarse_factor : math.ceil(py1 / coarse_factor),
                px0 // coarse_factor : math.ceil(px1 / coarse_factor),
            ]
            counts = np.bincount(block.ravel(), minlength=n_classes)[:n_classes]
            n_valid = int(np.count_nonzero(block != 255))
            if n_valid == 0 or n_valid / block.size < min_valid_frac:
                continue
            fg = 1.0 - counts[0] / n_valid
            if fg < label_threshold:
                continue
            y_est = _val_yield(
                px0, px1, py0, py1, width, height, patch_size, stride
            )
            if y_est == 0:
                continue
            mx0, my1 = _px_to_map(px0, py0, x0, y1, native_res)
            mx1, my0 = _px_to_map(px1, py1, x0, y1, native_res)
            cells.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "rect": [mx0, my0, mx1, my1],
                    "counts": counts.tolist(),
                    "yield": y_est,
                }
            )

    return {
        "crs": img_crs.to_string(),
        "total_patches": (
            len(window_origins(width, patch_size, stride))
            * len(window_origins(height, patch_size, stride))
        ),
        "total_class_counts": total_counts.tolist(),
        "cells": cells,
        "grid": [ncx, ncy],
    }


def _read_label(
    label_path,
    img_crs,
    attr_fields: str | Sequence[str] | None,
    attr_values: Sequence[int],
):
    """Return ``(bounds, geom_value_pairs)`` or ``(bounds, None)`` for rasters."""
    if check_label_type(label_path) == "raster":
        with rasterio.open(label_path) as lsrc:
            b = lsrc.bounds
            if lsrc.crs != img_crs:
                b = rasterio.warp.transform_bounds(lsrc.crs, img_crs, *b)
        return b, None

    gdf = load_vector_mask(label_path)
    field = resolve_attr_field(gdf.columns, attr_fields)
    if field is None:
        raise ValueError(f"{label_path}: none of {attr_fields} present")
    gdf = gdf[gdf[field].astype(float).astype(int).isin(list(attr_values))]
    if gdf.crs is not None and img_crs is not None and gdf.crs != img_crs:
        gdf = gdf.to_crs(img_crs)
    vals = gdf[field].astype(float).astype(int)
    order = np.argsort(vals.to_numpy(), kind="stable")
    geoms = [(gdf.geometry.iloc[i], int(vals.iloc[i])) for i in order]
    tb = gdf.total_bounds if len(gdf) else (0, 0, 0, 0)
    return tuple(tb), geoms


def _px_to_map(px, py, x_origin, y_origin_top, res):
    return x_origin + px * res, y_origin_top - py * res


def _val_yield(px0, px1, py0, py1, width, height, patch, stride):
    """Patch origins whose full window fits inside the cell."""

    def count(a0, a1, extent):
        return sum(
            1
            for o in window_origins(extent, patch, stride)
            if o >= a0 and o + patch <= a1
        )

    return count(px0, px1, width) * count(py0, py1, height)


def select_cells(
    candidates: list[dict],
    target: np.ndarray,
    quotas: dict[str, float],
    val_counts0: np.ndarray,
    spatial_eps: float = 1e-3,
) -> list[int]:
    """Deficit-aware greedy over candidate cells. Deterministic."""
    if not candidates:
        return []
    C = np.stack([c["counts"] for c in candidates]).astype(np.float64)
    Y = np.array([c["yield"] for c in candidates], dtype=np.float64)
    sensors = np.array([c["sensor"] for c in candidates])
    images = np.array([c["image"] for c in candidates])
    coords = np.array([[c["cx"], c["cy"]] for c in candidates], dtype=np.float64)

    remaining = dict(quotas)
    val = val_counts0.astype(np.float64).copy()
    selected: list[int] = []
    active = np.ones(len(candidates), dtype=bool)
    sel_by_image: dict[str, list[np.ndarray]] = {}

    while True:
        under = np.array([remaining.get(s, 0.0) > 0 for s in sensors])
        mask = active & under
        if not mask.any():
            break
        cand = np.where(mask)[0]
        after = val[None, :] + C[cand]
        dist = after / np.maximum(after.sum(axis=1, keepdims=True), 1e-9)
        gaps = np.abs(dist - target[None, :]).sum(axis=1)
        best = gaps.min()
        near = cand[gaps <= best + spatial_eps]
        pick = near[0]
        best_d = -1.0
        for i in near:
            prev = sel_by_image.get(images[i])
            d = (
                math.inf
                if not prev
                else min(float(np.max(np.abs(coords[i] - p))) for p in prev)
            )
            if d > best_d:
                best_d, pick = d, i
        selected.append(int(pick))
        active[pick] = False
        val += C[pick]
        remaining[sensors[pick]] -= Y[pick]
        sel_by_image.setdefault(images[pick], []).append(coords[pick])

    return selected


def run_planner(
    input_dict: list[dict],
    plan_file: str | Path,
    class_ids: dict[str, int],
    attr_fields: str | Sequence[str] | None,
    attr_values: Sequence[int] | None,
    patch_size: int = 512,
    stride: int = 256,
    val_ratio: float = 0.2,
    cell_strides: int = 4,
    coarse_factor: int = 2,
    label_threshold: float = 0.01,
    min_valid_frac: float = 0.5,
) -> dict:
    """Plan val cells across all sensors. tst pairs are ignored.

    Each item is ``{image, label, metadata}`` with ``split`` and
    ``collection`` (or ``sensor``) in metadata.
    """
    require_class_ids(class_ids)
    assign_pair_ids(input_dict)
    pairs = []
    for p in input_dict:
        meta = p.get("metadata") or {}
        split = require_split(meta, p.get("image"))
        if split == "trn" and p.get("label"):
            pairs.append(p)
    logger.info("Planning split for %d trn images", len(pairs))
    plan = load_plan(plan_file)
    plan["params"] = {
        "patch_size": patch_size,
        "stride": stride,
        "val_ratio": val_ratio,
        "cell_strides": cell_strides,
        "coarse_factor": coarse_factor,
        "label_threshold": label_threshold,
        "min_valid_frac": min_valid_frac,
        "class_ids": class_ids,
    }
    n_classes = max(class_ids.values()) + 1
    names = {v: k for k, v in class_ids.items()}

    analyses: dict[str, dict] = {}
    for p in pairs:
        name = p["metadata"]["pair_id"]
        meta = p.get("metadata") or {}
        sensor = meta.get("collection") or meta.get("sensor")
        if not sensor:
            raise ValueError(
                f"{p['image']}: metadata needs 'collection' or 'sensor'"
            )
        sensor = str(sensor)
        rec = plan["images"].get(name)
        if rec and rec.get("status") == "assigned":
            continue
        try:
            a = analyze_image(
                p["image"],
                p["label"],
                class_ids,
                attr_fields,
                attr_values,
                patch_size,
                stride,
                cell_strides,
                coarse_factor,
                label_threshold,
                min_valid_frac,
            )
        except Exception as e:
            logger.error("%s: analysis failed: %s", name, e)
            plan["images"][name] = {
                "sensor": sensor,
                "status": "error",
                "reason": str(e),
            }
            continue
        if a is None:
            plan["images"][name] = {"sensor": sensor, "status": "skipped"}
            continue
        a["sensor"] = sensor
        analyses[name] = a

    if not analyses:
        logger.info("No new images to assign.")
        save_plan(plan, plan_file)
        return plan

    total = np.zeros(n_classes)
    for rec in plan["images"].values():
        if rec.get("status") == "assigned":
            total += np.array(rec["total_class_counts"], dtype=np.float64)
    for a in analyses.values():
        total += np.array(a["total_class_counts"], dtype=np.float64)
    target = total / max(total.sum(), 1e-9)

    sensor_tot: dict[str, float] = {}
    sensor_val: dict[str, float] = {}
    for rec in plan["images"].values():
        if rec.get("status") == "assigned":
            s = rec["sensor"]
            sensor_tot[s] = sensor_tot.get(s, 0) + rec["total_patches"]
            sensor_val[s] = sensor_val.get(s, 0) + rec["est_val_patches"]
    for a in analyses.values():
        s = a["sensor"]
        sensor_tot[s] = sensor_tot.get(s, 0) + a["total_patches"]
    quotas = {
        s: max(0.0, val_ratio * t - sensor_val.get(s, 0.0))
        for s, t in sensor_tot.items()
    }

    val0 = np.zeros(n_classes)
    for rec in plan["images"].values():
        if rec.get("status") == "assigned":
            val0 += np.array(rec["val_class_counts"], dtype=np.float64)

    candidates = []
    for name, a in analyses.items():
        for c in a["cells"]:
            candidates.append(
                {
                    "image": name,
                    "sensor": a["sensor"],
                    "cx": c["cx"],
                    "cy": c["cy"],
                    "counts": np.array(c["counts"], dtype=np.float64),
                    "yield": c["yield"],
                    "rect": c["rect"],
                }
            )
    chosen = select_cells(candidates, target, quotas, val0)
    chosen_by_image: dict[str, list[dict]] = {}
    for i in chosen:
        chosen_by_image.setdefault(candidates[i]["image"], []).append(candidates[i])

    for name, a in analyses.items():
        picked = chosen_by_image.get(name, [])
        vc = np.zeros(n_classes)
        for c in picked:
            vc += c["counts"]
        plan["images"][name] = {
            "sensor": a["sensor"],
            "crs": a["crs"],
            "status": "assigned",
            "val_cells": merge_adjacent_rects([c["rect"] for c in picked]),
            "est_val_patches": int(sum(c["yield"] for c in picked)),
            "total_patches": int(a["total_patches"]),
            "total_class_counts": a["total_class_counts"],
            "val_class_counts": vc.tolist(),
            "assigned_at": _now(),
        }

    total_cc = np.zeros(n_classes)
    val_cc = np.zeros(n_classes)
    plan["sensors"] = {}
    for rec in plan["images"].values():
        if rec.get("status") != "assigned":
            continue
        s = rec["sensor"]
        d = plan["sensors"].setdefault(
            s, {"total_patches": 0, "est_val_patches": 0}
        )
        d["total_patches"] += rec["total_patches"]
        d["est_val_patches"] += rec["est_val_patches"]
        total_cc += np.array(rec["total_class_counts"])
        val_cc += np.array(rec["val_class_counts"])
    plan["target_distribution"] = {
        names[i]: float(v)
        for i, v in enumerate(total_cc / max(total_cc.sum(), 1e-9))
    }
    plan["global"] = {
        "total_class_counts": {
            names[i]: float(v) for i, v in enumerate(total_cc)
        },
        "val_class_counts": {names[i]: float(v) for i, v in enumerate(val_cc)},
        "val_distribution": {
            names[i]: float(v)
            for i, v in enumerate(val_cc / max(val_cc.sum(), 1e-9))
        },
    }
    save_plan(plan, plan_file)
    _log_summary(plan)
    return plan


def _log_summary(plan: dict) -> None:
    logger.info("=== split plan summary ===")
    for s, d in sorted(plan["sensors"].items()):
        r = d["est_val_patches"] / max(d["total_patches"], 1)
        logger.info(
            "  %-16s patches=%6d  val_est=%5d  (%.1f%%)",
            s,
            d["total_patches"],
            d["est_val_patches"],
            100 * r,
        )
    t = plan["target_distribution"]
    v = plan["global"]["val_distribution"]
    for cls in t:
        logger.info("  class %-10s target=%.4f  val=%.4f", cls, t[cls], v[cls])


def merge_adjacent_rects(rects: list, tol: float = 1e-6) -> list:
    """Merge axis-aligned rects that share a full edge (fixpoint).

    Patches spanning two adjacent val cells then classify as inside the union.
    """
    rects = [list(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(rects)
        for i, a in enumerate(rects):
            if used[i]:
                continue
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                b = rects[j]
                same_y = abs(a[1] - b[1]) < tol and abs(a[3] - b[3]) < tol
                same_x = abs(a[0] - b[0]) < tol and abs(a[2] - b[2]) < tol
                touch_x = abs(a[2] - b[0]) < tol or abs(b[2] - a[0]) < tol
                touch_y = abs(a[3] - b[1]) < tol or abs(b[3] - a[1]) < tol
                if (same_y and touch_x) or (same_x and touch_y):
                    a = [
                        min(a[0], b[0]),
                        min(a[1], b[1]),
                        max(a[2], b[2]),
                        max(a[3], b[3]),
                    ]
                    used[j] = True
                    changed = True
            out.append(a)
            used[i] = True
        rects = out
    return rects


def val_rects_for_image(plan: dict, image_name: str, dst_crs) -> list | None:
    """Val rectangles in ``dst_crs``. None if the image has no assignment."""
    rec = plan["images"].get(image_name)
    if not rec or rec.get("status") != "assigned":
        return None
    rects = rec["val_cells"]
    src_crs = rasterio.crs.CRS.from_string(rec["crs"])
    if dst_crs is not None and src_crs != dst_crs:
        rects = [
            list(rasterio.warp.transform_bounds(src_crs, dst_crs, *r))
            for r in rects
        ]
    return rects


def classify_patch(
    bounds, val_rects, tol: float, img_bounds=None
) -> str | None:
    """``val`` if inside a val rect, ``trn`` if disjoint, else None (buffer).

    ``bounds`` / rects are ``(minx, miny, maxx, maxy)``. ``tol`` is in map units
    (~1 native pixel). ``img_bounds`` clips window bounds to the scene.
    """
    x0, y0, x1, y1 = bounds
    if img_bounds is not None:
        x0 = max(x0, img_bounds[0])
        y0 = max(y0, img_bounds[1])
        x1 = min(x1, img_bounds[2])
        y1 = min(y1, img_bounds[3])
    inside = intersects = False
    for rx0, ry0, rx1, ry1 in val_rects:
        if (
            x0 >= rx0 - tol
            and y0 >= ry0 - tol
            and x1 <= rx1 + tol
            and y1 <= ry1 + tol
        ):
            inside = True
            break
        if (
            x0 < rx1 - tol
            and x1 > rx0 + tol
            and y0 < ry1 - tol
            and y1 > ry0 + tol
        ):
            intersects = True
    if inside:
        return "val"
    return None if intersects else "trn"
