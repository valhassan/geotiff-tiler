"""Pre-flight checks for Tiler image/label pairs."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import box
from tqdm import tqdm

from geotiff_tiler.lib.geo import check_label_type, check_label_validity
from geotiff_tiler.lib.io import load_vector_mask, resolve_attr_field, validate_image

logger = logging.getLogger(__name__)

HARD_CHECKS = frozenset(
    {
        "image_readable",
        "band_count",
        "not_degenerate",
        "label_valid",
        "spatial_overlap",
        "attr_field_exists",
    }
)
SOFT_CHECKS = frozenset({"crs_match", "attr_values_match", "nonempty_after_filter"})
LABEL_CHECKS = (
    "label_valid",
    "crs_match",
    "spatial_overlap",
    "attr_field_exists",
    "attr_values_match",
    "nonempty_after_filter",
)

SAMPLE_MAX_DIM = 256


@dataclass
class VerificationResult:
    """Rollup for a single pair.

    Attributes:
        id: Pair identifier.
        image: Original image path or STAC href.
        label: Label path, or None for image-only.
        status: ``ok``, ``warning``, or ``error``.
        checks: Per-check ``{passed, detail}`` map.
        errors: Exception messages from hard failures.
    """

    id: str
    image: str
    label: str | None
    status: str
    checks: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _check(passed: bool, detail: Any = None) -> dict:
    return {"passed": bool(passed), "detail": detail}


def _skipped(reason: str) -> dict:
    return {"passed": True, "detail": {"skipped": True, "reason": reason}}


def _skip_remaining(checks: dict[str, dict], names: Sequence[str], reason: str) -> None:
    for name in names:
        if name not in checks:
            checks[name] = _skipped(reason)


def _overlap_fractions(poly_a, poly_b) -> dict[str, float]:
    """Containment both ways plus IoU."""
    area_a, area_b = poly_a.area, poly_b.area
    inter = poly_a.intersection(poly_b)
    inter_area = 0.0 if inter.is_empty else inter.area
    union_area = poly_a.union(poly_b).area
    return {
        "overlap_a_rto_b": (inter_area / area_a) if area_a > 0 else 0.0,
        "overlap_b_rto_a": (inter_area / area_b) if area_b > 0 else 0.0,
        "iou": (inter_area / union_area) if union_area > 0 else 0.0,
        "intersection_empty": inter.is_empty or inter_area == 0.0,
    }


def _vector_bounds(gdf: gpd.GeoDataFrame):
    if hasattr(gdf, "attrs") and "extent_geometry" in gdf.attrs:
        return gdf.attrs["extent_geometry"]
    return box(*gdf.total_bounds)


def _overview_hw(src: rasterio.DatasetReader) -> tuple[int, int]:
    return (
        max(1, min(SAMPLE_MAX_DIM, src.height)),
        max(1, min(SAMPLE_MAX_DIM, src.width)),
    )


def _sample_is_degenerate(src: rasterio.DatasetReader) -> tuple[bool, dict]:
    h, w = _overview_hw(src)
    data = src.read(out_shape=(src.count, h, w), resampling=Resampling.nearest)
    nodata = src.nodata
    all_zero = bool(np.all(data == 0))
    all_nodata = bool(nodata is not None and np.all(data == nodata))
    return all_zero or all_nodata, {
        "sample_shape": [src.count, h, w],
        "all_zero": all_zero,
        "all_nodata": all_nodata,
        "nodata": nodata,
    }


def _can_int(v: Any) -> bool:
    try:
        if isinstance(v, bool):
            return False
        int(v)
        return True
    except (TypeError, ValueError):
        return False


def _norm_set(values: Sequence[Any]) -> set:
    """Int/str-tolerant set for attr/class comparisons."""
    out: set = set()
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out.add(v)
        out.add(str(v))
        if _can_int(v):
            iv = int(v)
            out.add(iv)
            out.add(str(iv))
    return out


def _rollup_status(checks: dict[str, dict], errors: list[str]) -> str:
    if errors or any(
        name in HARD_CHECKS and not c.get("passed", False)
        for name, c in checks.items()
    ):
        return "error"
    if any(
        name in SOFT_CHECKS and not c.get("passed", False)
        for name, c in checks.items()
    ):
        return "warning"
    return "ok"


def _finish(
    result_id: str,
    image: str,
    label: str | None,
    checks: dict[str, dict],
    errors: list[str],
) -> VerificationResult:
    return VerificationResult(
        id=result_id,
        image=str(image),
        label=None if label is None else str(label),
        status=_rollup_status(checks, errors),
        checks=checks,
        errors=errors,
    )


def verify_pair(
    image: str,
    label: str | None = None,
    *,
    attr_field: str | Sequence[str] | None = None,
    attr_values: list | None = None,
    class_ids: dict | None = None,
    bands_expected: int | None = None,
    bands_requested: Sequence[str] | None = None,
    band_indices: Sequence[int] | None = None,
    pair_id: str | None = None,
) -> VerificationResult:
    """Run all verification checks on one image/label pair.

    Args:
        image: Path or STAC item href.
        label: Raster/vector label path, or None for image-only.
        attr_field: Vector attribute field name(s).
        attr_values: Declared values for ``attr_field``.
        class_ids: Name-to-id mapping for raster labels.
        bands_expected: Required image band count, if known.
        bands_requested: STAC common-name bands for ``validate_image``.
        band_indices: 1-based source band order for local TIFFs.
        pair_id: Report id; defaults to the image stem.

    Returns:
        VerificationResult with per-check details and a rollup status.
    """
    result_id = pair_id or Path(str(image)).stem
    checks: dict[str, dict] = {}
    errors: list[str] = []

    image_crs = None
    image_bounds = None
    band_count = None
    degenerate = False
    deg_detail: dict | None = None

    try:
        vkw: dict[str, Any] = {"band_indices": band_indices}
        if bands_requested is not None:
            vkw["bands_requested"] = bands_requested
        with tempfile.TemporaryDirectory() as td:
            vkw["tmp_dir"] = td
            resolved = validate_image(image, **vkw)
            with rasterio.open(resolved) as src:
                image_crs = src.crs
                image_bounds = box(*src.bounds)
                band_count = src.count
                degenerate, deg_detail = _sample_is_degenerate(src)
        checks["image_readable"] = _check(
            True,
            {"opened_via": "stac_vrt" if str(resolved) != str(image) else "path"},
        )
    except Exception as e:
        errors.append(f"image_readable: {e}")
        checks["image_readable"] = _check(False, str(e))
        checks["band_count"] = _skipped("image unreadable")
        checks["not_degenerate"] = _skipped("image unreadable")

    if "band_count" not in checks:
        if bands_expected is None:
            checks["band_count"] = _check(
                True, {"count": band_count, "expected": None, "skipped": True}
            )
        else:
            checks["band_count"] = _check(
                band_count == bands_expected,
                {"count": band_count, "expected": bands_expected},
            )

    if "not_degenerate" not in checks:
        checks["not_degenerate"] = _check(not degenerate, deg_detail)

    if label is None:
        _skip_remaining(checks, LABEL_CHECKS, "inference-only (label is None)")
        return _finish(result_id, image, None, checks, errors)

    label_type: str | None = None
    label_gdf: gpd.GeoDataFrame | None = None
    label_crs = None
    label_bounds = None
    raster_sample: np.ndarray | None = None
    raster_nodata = None

    try:
        label_type = check_label_type(label)
        if label_type == "vector":
            label_gdf = load_vector_mask(label)
            valid, msg = check_label_validity(label_gdf)
            detail: dict[str, Any] = {
                "type": "vector",
                "n_features": len(label_gdf),
                "msg": msg,
            }
            if not label_gdf.empty:
                n_invalid = int((~label_gdf.geometry.is_valid).sum())
                if n_invalid:
                    detail["n_invalid"] = n_invalid
            checks["label_valid"] = _check(valid, detail)
            label_crs = label_gdf.crs
            label_bounds = _vector_bounds(label_gdf)
        else:
            with rasterio.open(label) as src_label:
                valid, msg = check_label_validity(src_label)
                label_crs = src_label.crs
                label_bounds = box(*src_label.bounds)
                h, w = _overview_hw(src_label)
                raster_sample = src_label.read(
                    1, out_shape=(h, w), resampling=Resampling.nearest
                )
                raster_nodata = src_label.nodata
                checks["label_valid"] = _check(
                    valid,
                    {
                        "type": "raster",
                        "width": src_label.width,
                        "height": src_label.height,
                        "msg": msg,
                    },
                )
    except Exception as e:
        errors.append(f"label_valid: {e}")
        checks["label_valid"] = _check(False, str(e))
        _skip_remaining(checks, LABEL_CHECKS, "label unreadable")
        return _finish(result_id, image, label, checks, errors)

    if image_bounds is None:
        checks["crs_match"] = _skipped("image unreadable")
    elif image_crs is None or label_crs is None:
        checks["crs_match"] = _check(
            False,
            {"image_crs": str(image_crs), "label_crs": str(label_crs)},
        )
    else:
        checks["crs_match"] = _check(
            image_crs == label_crs,
            {"image_crs": str(image_crs), "label_crs": str(label_crs)},
        )

    try:
        if image_bounds is None:
            checks["spatial_overlap"] = _skipped("image unreadable")
        else:
            fracs = _overlap_fractions(label_bounds, image_bounds)
            checks["spatial_overlap"] = _check(
                not fracs["intersection_empty"],
                {
                    "overlap_label_rto_raster": fracs["overlap_a_rto_b"],
                    "overlap_raster_rto_label": fracs["overlap_b_rto_a"],
                    "iou": fracs["iou"],
                },
            )
    except Exception as e:
        errors.append(f"spatial_overlap: {e}")
        checks["spatial_overlap"] = _check(False, str(e))

    _run_attr_checks(
        checks,
        errors,
        label_type=label_type,
        label_gdf=label_gdf,
        raster_sample=raster_sample,
        raster_nodata=raster_nodata,
        attr_field=attr_field,
        attr_values=attr_values,
        class_ids=class_ids,
    )
    return _finish(result_id, image, label, checks, errors)


def _run_attr_checks(
    checks: dict[str, dict],
    errors: list[str],
    *,
    label_type: str,
    label_gdf: gpd.GeoDataFrame | None,
    raster_sample: np.ndarray | None,
    raster_nodata: Any,
    attr_field: str | Sequence[str] | None,
    attr_values: list | None,
    class_ids: dict | None,
) -> None:
    resolved: str | None = None
    if label_type != "vector" or attr_field is None:
        checks["attr_field_exists"] = _skipped(
            "not applicable (raster label or no attr_field)"
        )
    else:
        try:
            assert label_gdf is not None
            resolved = resolve_attr_field(label_gdf.columns, attr_field)
            checks["attr_field_exists"] = _check(
                resolved is not None,
                {
                    "requested": (
                        attr_field
                        if isinstance(attr_field, str)
                        else list(attr_field)
                    ),
                    "resolved": resolved,
                },
            )
        except Exception as e:
            errors.append(f"attr_field_exists: {e}")
            checks["attr_field_exists"] = _check(False, str(e))

    try:
        if label_type == "vector":
            if attr_values is None:
                checks["attr_values_match"] = _skipped("no attr_values declared")
            elif attr_field is None:
                checks["attr_values_match"] = _skipped("no attr_field declared")
            elif resolved is None:
                checks["attr_values_match"] = _skipped("attr_field missing")
            else:
                assert label_gdf is not None
                present = list(label_gdf[resolved].dropna().unique())
                declared = list(attr_values)
                present_n, declared_n = _norm_set(present), _norm_set(declared)
                extra = [v for v in present if v not in declared_n]
                missing = [v for v in declared if v not in present_n]
                checks["attr_values_match"] = _check(
                    not extra and not missing,
                    {
                        "present": sorted(present, key=str),
                        "declared": list(attr_values),
                        "extra_in_data": extra,
                        "missing_from_data": missing,
                    },
                )
        elif not class_ids:
            checks["attr_values_match"] = _skipped("no class_ids declared")
        else:
            declared = set(class_ids.values())
            present = set(np.unique(raster_sample).tolist())
            if raster_nodata is not None:
                present.discard(raster_nodata)
            extra = sorted(present - declared, key=str)
            missing = sorted(declared - present, key=str)
            # Sample may miss rare classes; fail only on undeclared values.
            checks["attr_values_match"] = _check(
                not extra,
                {
                    "present": sorted(present, key=str),
                    "declared": sorted(declared, key=str),
                    "extra_in_data": extra,
                    "missing_from_sample": missing,
                },
            )
    except Exception as e:
        errors.append(f"attr_values_match: {e}")
        checks["attr_values_match"] = _check(False, str(e))

    try:
        if label_type == "vector":
            if attr_field is None or attr_values is None:
                checks["nonempty_after_filter"] = _skipped(
                    "no attr_field/attr_values filter"
                )
            elif resolved is None:
                checks["nonempty_after_filter"] = _check(
                    False, {"empty_after_filter": True, "n": 0}
                )
            else:
                assert label_gdf is not None
                declared_n = _norm_set(attr_values)
                n = int(label_gdf[resolved].apply(lambda v: v in declared_n).sum())
                checks["nonempty_after_filter"] = _check(
                    n > 0, {"empty_after_filter": n == 0, "n": n}
                )
        elif not class_ids:
            checks["nonempty_after_filter"] = _skipped("no class_ids declared")
        else:
            fg = {v for v in class_ids.values() if v != 0} or set(class_ids.values())
            n = int(np.isin(raster_sample, list(fg)).sum())
            checks["nonempty_after_filter"] = _check(
                n > 0, {"empty_after_filter": n == 0, "n_labeled_sample": n}
            )
    except Exception as e:
        errors.append(f"nonempty_after_filter: {e}")
        checks["nonempty_after_filter"] = _check(False, str(e))


def _flatten_result(result: VerificationResult) -> dict:
    row: dict[str, Any] = {
        "id": result.id,
        "image": result.image,
        "label": result.label,
        "status": result.status,
        "errors": json.dumps(result.errors),
    }
    for name, check in result.checks.items():
        row[f"check_{name}_passed"] = check.get("passed")
        detail = check.get("detail")
        row[f"check_{name}_detail"] = (
            json.dumps(detail, default=str)
            if isinstance(detail, (dict, list))
            else detail
        )
    return row


def verify_dataset(
    input_dict: list[dict],
    output_report_path: str | None = None,
    *,
    attr_field: str | Sequence[str] | None = None,
    attr_values: list | None = None,
    class_ids: dict | None = None,
    bands_expected: int | None = None,
    bands_requested: Sequence[str] | None = None,
    band_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Verify all pairs in a Tiler-style ``input_dict``.

    Args:
        input_dict: List of ``{image, label, metadata}`` dicts.
        output_report_path: Optional CSV path. The in-memory frame also
            keeps a structured ``checks`` column.
        attr_field: Vector attribute field name(s).
        attr_values: Declared values for ``attr_field``.
        class_ids: Name-to-id mapping for raster labels.
        bands_expected: Required image band count, if known.
        bands_requested: STAC common-name bands for ``validate_image``.
        band_indices: 1-based source band order for local TIFFs.

    Returns:
        DataFrame with one row per pair and a rollup logged to INFO.
    """
    results: list[VerificationResult] = []
    for item in tqdm(input_dict, desc="Verifying pairs"):
        meta = item.get("metadata") or {}
        pair_id = meta.get("id") or meta.get("aoi_id") or Path(str(item["image"])).stem
        results.append(
            verify_pair(
                image=item["image"],
                label=item.get("label"),
                pair_id=str(pair_id),
                attr_field=attr_field,
                attr_values=attr_values,
                class_ids=class_ids,
                bands_expected=bands_expected,
                bands_requested=bands_requested,
                band_indices=band_indices,
            )
        )

    df = pd.DataFrame([_flatten_result(r) for r in results])
    df["checks"] = [r.checks for r in results]

    if output_report_path:
        out = Path(output_report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["checks"]).to_csv(out, index=False)
        logger.info("Wrote verification report to %s", out)

    counts = df["status"].value_counts() if len(df) else pd.Series(dtype=int)
    logger.info(
        "Verification complete: %d ok, %d warning, %d error (total %d)",
        int(counts.get("ok", 0)),
        int(counts.get("warning", 0)),
        int(counts.get("error", 0)),
        len(df),
    )
    return df
