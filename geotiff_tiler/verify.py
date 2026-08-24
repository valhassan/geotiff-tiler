"""Pre-flight checks for Tiler image/label pairs."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
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

from geotiff_tiler.utils.checks import check_label_validity
from geotiff_tiler.utils.io import load_vector_mask, validate_image

logger = logging.getLogger(__name__)

HARD_CHECKS = frozenset(
    {
        "image_readable",
        "band_count",
        "not_degenerate",
        "label_valid",
        "crs_match",
        "spatial_overlap",
        "attr_field_exists",
    }
)
SOFT_CHECKS = frozenset({"attr_values_match", "nonempty_after_filter"})
LABEL_CHECKS = (
    "label_valid",
    "crs_match",
    "spatial_overlap",
    "attr_field_exists",
    "attr_values_match",
    "nonempty_after_filter",
)

SAMPLE_MAX_DIM = 256
VECTOR_EXTS = (".geojson", ".gpkg", ".shp")
RASTER_EXTS = (".tif", ".tiff")
_CSV_RESERVED = {"image", "label"}


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


def _label_type(label_path: str) -> str:
    lower = label_path.lower()
    if lower.endswith(RASTER_EXTS):
        return "raster"
    if lower.endswith(VECTOR_EXTS):
        return "vector"
    raise ValueError(
        f"Invalid label type: {label_path}, "
        "must be raster (.tif/.tiff) or vector (.geojson/.gpkg/.shp)"
    )


def _resolve_attr_field(
    columns: Sequence[str], attr_field: str | Sequence[str] | None
) -> str | None:
    if attr_field is None:
        return None
    fields = [attr_field] if isinstance(attr_field, str) else list(attr_field)
    for name in fields:
        if name in columns:
            return name
    return None


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
        resolved = (
            validate_image(image, bands_requested=bands_requested)
            if bands_requested is not None
            else validate_image(image)
        )
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
        label_type = _label_type(label)
        if label_type == "vector":
            label_gdf = (
                load_vector_mask(label)
                if Path(label).suffix.lower() == ".gpkg"
                else gpd.read_file(label)
            )
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

    try:
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
    except Exception as e:
        errors.append(f"crs_match: {e}")
        checks["crs_match"] = _check(False, str(e))

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
            resolved = _resolve_attr_field(label_gdf.columns, attr_field)
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


def _load_input_dict(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of {image, label, metadata}")
        return data

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if "image" not in df.columns:
            raise ValueError(
                f"CSV must include an 'image' column, got {list(df.columns)}"
            )
        rows = []
        for rec in df.to_dict(orient="records"):
            label = rec.get("label")
            if label is not None and pd.isna(label):
                label = None
            meta = {
                k: v for k, v in rec.items() if k not in _CSV_RESERVED and pd.notna(v)
            }
            rows.append({"image": rec["image"], "label": label, "metadata": meta})
        return rows

    raise ValueError(f"Unsupported input format: {p.suffix} (use .json or .csv)")


def _parse_class_ids(raw: str | None) -> dict | None:
    if raw is None:
        return None
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--class_ids must be a dict literal")
    return parsed


def _parse_attr_values(raw: list[str] | None) -> list | None:
    if raw is None:
        return None
    out = []
    for v in raw:
        try:
            out.append(int(v))
        except ValueError:
            out.append(v)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m geotiff_tiler.verify",
        description="Pre-flight verify image/label pairs before tiling.",
    )
    parser.add_argument("input", help="JSON list or CSV with an image column")
    parser.add_argument(
        "-o", "--output", default="verification_report.csv", help="Output CSV path"
    )
    parser.add_argument("--attr_field", nargs="+", default=None)
    parser.add_argument("--attr_values", nargs="+", default=None)
    parser.add_argument(
        "--class_ids",
        default=None,
        help="Dict literal, e.g. \"{'background': 0, 'fore': 1}\"",
    )
    parser.add_argument("--bands_expected", type=int, default=None)
    parser.add_argument(
        "--bands_requested",
        nargs="+",
        default=None,
        help="STAC common-name bands (default: validate_image default)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    attr_field: str | list[str] | None = args.attr_field
    if attr_field is not None and len(attr_field) == 1:
        attr_field = attr_field[0]

    df = verify_dataset(
        _load_input_dict(args.input),
        output_report_path=args.output,
        attr_field=attr_field,
        attr_values=_parse_attr_values(args.attr_values),
        class_ids=_parse_class_ids(args.class_ids),
        bands_expected=args.bands_expected,
        bands_requested=args.bands_requested,
    )
    n_err = int((df["status"] == "error").sum()) if len(df) else 0
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
