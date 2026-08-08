"""Standalone pre-flight verification for Tiler input_dict pairs.

Answers: is this image/label pair safe to tile, and if not, why?
No Hydra, no AOI — opens files with rasterio/geopandas and emits one report row
per pair.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

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

SAMPLE_MAX_DIM = 256
VECTOR_EXTS = (".geojson", ".gpkg", ".shp")
RASTER_EXTS = (".tif", ".tiff")


@dataclass
class VerificationResult:
    id: str
    image: str
    label: str | None
    status: str  # ok | warning | error
    checks: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _check(passed: bool, detail: Any = None) -> dict:
    return {"passed": bool(passed), "detail": detail}


def _skipped(reason: str) -> dict:
    return {"passed": True, "detail": {"skipped": True, "reason": reason}}


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
    columns: Sequence[str], attr_field: Union[str, Sequence[str], None]
) -> Optional[str]:
    if attr_field is None:
        return None
    fields = [attr_field] if isinstance(attr_field, str) else list(attr_field)
    candidates = [f for f in fields if f in columns]
    if not candidates:
        return None
    return candidates[0]


def _overlap_fractions(poly_a, poly_b) -> dict[str, float]:
    """Return containment fractions both ways plus IoU."""
    area_a = poly_a.area
    area_b = poly_b.area
    inter = poly_a.intersection(poly_b)
    inter_area = 0.0 if inter.is_empty else inter.area
    union_area = poly_a.union(poly_b).area
    return {
        "overlap_a_rto_b": (inter_area / area_a) if area_a > 0 else 0.0,
        "overlap_b_rto_a": (inter_area / area_b) if area_b > 0 else 0.0,
        "iou": (inter_area / union_area) if union_area > 0 else 0.0,
        "intersection_empty": inter.is_empty or inter_area == 0.0,
    }


def _label_bounds(label_path: str, label_type: str, gdf: gpd.GeoDataFrame | None):
    if label_type == "raster":
        with rasterio.open(label_path) as src:
            return box(*src.bounds), src.crs
    assert gdf is not None
    if hasattr(gdf, "attrs") and "extent_geometry" in gdf.attrs:
        bounds_geom = gdf.attrs["extent_geometry"]
    else:
        bounds_geom = box(*gdf.total_bounds)
    return bounds_geom, gdf.crs


def _sample_is_degenerate(src: rasterio.DatasetReader) -> tuple[bool, dict]:
    """Cheap degeneracy check via decimated overview-style read."""
    h = max(1, min(SAMPLE_MAX_DIM, src.height))
    w = max(1, min(SAMPLE_MAX_DIM, src.width))
    out_shape = (src.count, h, w)
    data = src.read(out_shape=out_shape, resampling=Resampling.nearest)
    nodata = src.nodata

    all_zero = bool(np.all(data == 0))
    if nodata is None:
        all_nodata = False
    else:
        all_nodata = bool(np.all(data == nodata))

    degenerate = all_zero or all_nodata
    return degenerate, {
        "sample_shape": list(out_shape),
        "all_zero": all_zero,
        "all_nodata": all_nodata,
        "nodata": nodata,
    }


def _normalize_values(values: Sequence[Any]) -> set:
    """Normalize attr/class values for set comparison (int/str tolerant)."""
    out: set = set()
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out.add(v)
        try:
            if not isinstance(v, bool):
                out.add(int(v))
                out.add(str(int(v)))
        except (TypeError, ValueError):
            pass
        out.add(str(v))
    return out


def _rollup_status(checks: dict[str, dict], errors: list[str]) -> str:
    if errors:
        return "error"
    hard_fail = any(
        name in HARD_CHECKS and not c.get("passed", False)
        for name, c in checks.items()
    )
    if hard_fail:
        return "error"
    soft_fail = any(
        name in SOFT_CHECKS and not c.get("passed", False)
        for name, c in checks.items()
    )
    if soft_fail:
        return "warning"
    return "ok"


def verify_pair(
    image: str,
    label: str | None = None,
    *,
    attr_field: Union[str, Sequence[str], None] = None,
    attr_values: list | None = None,
    class_ids: dict | None = None,
    bands_expected: int | None = None,
    bands_requested: Sequence[str] | None = None,
    pair_id: str | None = None,
) -> VerificationResult:
    """Run all verification checks on a single image/label pair."""
    result_id = pair_id or Path(str(image)).stem
    checks: dict[str, dict] = {}
    errors: list[str] = []

    image_path: str | None = None
    label_type: str | None = None
    label_gdf: gpd.GeoDataFrame | None = None
    image_crs = None
    label_crs = None

    # --- 1. image_readable ---
    # Keep original source in the report (STAC URL / path). Do NOT dump the
    # in-memory VRT XML that validate_image returns for STAC items.
    try:
        resolved = validate_image(
            image, bands_requested=bands_requested or ["red", "green", "blue"]
        )
        with rasterio.open(resolved) as src:
            _ = src.meta
            image_path = str(resolved)
            image_crs = src.crs
            band_count = src.count
            # Peek for later checks while open
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
        # Still attempt label-only checks below where possible
        band_count = None
        degenerate = None
        deg_detail = None

    # --- 2. band_count ---
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

    # --- 3. not_degenerate ---
    if "not_degenerate" not in checks:
        checks["not_degenerate"] = _check(not degenerate, deg_detail)

    # --- label path ---
    if label is None:
        for name in (
            "label_valid",
            "crs_match",
            "spatial_overlap",
            "attr_field_exists",
            "attr_values_match",
            "nonempty_after_filter",
        ):
            checks[name] = _skipped("inference-only (label is None)")
        status = _rollup_status(checks, errors)
        return VerificationResult(
            id=result_id,
            image=str(image),
            label=None,
            status=status,
            checks=checks,
            errors=errors,
        )

    # --- 4. label_valid ---
    try:
        label_type = _label_type(label)
        if label_type == "vector":
            if Path(label).suffix.lower() == ".gpkg":
                label_gdf = load_vector_mask(label)
            else:
                label_gdf = gpd.read_file(label)
            valid, msg = check_label_validity(label_gdf)
            # Report invalid geoms even if load_vector_mask auto-fixed gpkg
            if label_gdf is not None and not label_gdf.empty:
                n_invalid = int((~label_gdf.geometry.is_valid).sum())
                detail = {"type": "vector", "n_features": len(label_gdf), "msg": msg}
                if n_invalid:
                    detail["n_invalid"] = n_invalid
                    valid = False
                checks["label_valid"] = _check(valid, detail)
            else:
                checks["label_valid"] = _check(valid, {"type": "vector", "msg": msg})
            label_crs = label_gdf.crs if label_gdf is not None else None
        else:
            with rasterio.open(label) as src_label:
                valid, msg = check_label_validity(src_label)
                label_crs = src_label.crs
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
        for name in (
            "crs_match",
            "spatial_overlap",
            "attr_field_exists",
            "attr_values_match",
            "nonempty_after_filter",
        ):
            checks[name] = _skipped("label unreadable")
        status = _rollup_status(checks, errors)
        return VerificationResult(
            id=result_id,
            image=str(image),
            label=str(label),
            status=status,
            checks=checks,
            errors=errors,
        )

    # --- 5. crs_match ---
    try:
        if image_path is None:
            checks["crs_match"] = _skipped("image unreadable")
        elif image_crs is None or label_crs is None:
            checks["crs_match"] = _check(
                False,
                {"image_crs": str(image_crs), "label_crs": str(label_crs)},
            )
        else:
            match = image_crs == label_crs
            checks["crs_match"] = _check(
                match,
                {"image_crs": str(image_crs), "label_crs": str(label_crs)},
            )
    except Exception as e:
        errors.append(f"crs_match: {e}")
        checks["crs_match"] = _check(False, str(e))

    # --- 6. spatial_overlap ---
    try:
        if image_path is None:
            checks["spatial_overlap"] = _skipped("image unreadable")
        else:
            with rasterio.open(image_path) as src:
                image_bounds = box(*src.bounds)
            label_bounds, _ = _label_bounds(label, label_type, label_gdf)
            fracs = _overlap_fractions(label_bounds, image_bounds)
            detail = {
                "overlap_label_rto_raster": fracs["overlap_a_rto_b"],
                "overlap_raster_rto_label": fracs["overlap_b_rto_a"],
                "iou": fracs["iou"],
            }
            checks["spatial_overlap"] = _check(
                not fracs["intersection_empty"], detail
            )
    except Exception as e:
        errors.append(f"spatial_overlap: {e}")
        checks["spatial_overlap"] = _check(False, str(e))

    # --- 7–9 attribute / class checks ---
    _run_attr_checks(
        checks,
        errors,
        label=label,
        label_type=label_type,
        label_gdf=label_gdf,
        attr_field=attr_field,
        attr_values=attr_values,
        class_ids=class_ids,
    )

    status = _rollup_status(checks, errors)
    return VerificationResult(
        id=result_id,
        image=str(image),
        label=str(label),
        status=status,
        checks=checks,
        errors=errors,
    )


def _run_attr_checks(
    checks: dict[str, dict],
    errors: list[str],
    *,
    label: str,
    label_type: str,
    label_gdf: gpd.GeoDataFrame | None,
    attr_field: Union[str, Sequence[str], None],
    attr_values: list | None,
    class_ids: dict | None,
) -> None:
    # --- 7. attr_field_exists (vector + attr_field only) ---
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
                    "columns": list(map(str, label_gdf.columns)),
                },
            )
        except Exception as e:
            errors.append(f"attr_field_exists: {e}")
            checks["attr_field_exists"] = _check(False, str(e))

    # --- 8. attr_values_match ---
    try:
        if label_type == "vector":
            if attr_values is None:
                checks["attr_values_match"] = _skipped("no attr_values declared")
            elif attr_field is None:
                checks["attr_values_match"] = _skipped("no attr_field declared")
            else:
                assert label_gdf is not None
                resolved = _resolve_attr_field(label_gdf.columns, attr_field)
                if resolved is None:
                    checks["attr_values_match"] = _skipped("attr_field missing")
                else:
                    present = set(label_gdf[resolved].dropna().unique().tolist())
                    declared = set(attr_values)
                    # Compare with int/str tolerance
                    present_n = _normalize_values(present)
                    declared_n = _normalize_values(declared)
                    extra = [
                        v
                        for v in present
                        if v not in declared_n and str(v) not in declared_n
                    ]
                    missing = [
                        v
                        for v in declared
                        if v not in present_n and str(v) not in present_n
                    ]
                    # Also try int coercion for missing
                    missing = [
                        v
                        for v in missing
                        if not (
                            (isinstance(v, (int, str)) and int(v) in present_n)
                            if _can_int(v)
                            else False
                        )
                    ]
                    extra = [
                        v
                        for v in extra
                        if not (
                            (_can_int(v) and int(v) in declared_n)
                            or str(v) in {str(d) for d in declared}
                        )
                    ]
                    checks["attr_values_match"] = _check(
                        len(extra) == 0 and len(missing) == 0,
                        {
                            "present": sorted(present, key=str),
                            "declared": list(attr_values),
                            "extra_in_data": extra,
                            "missing_from_data": missing,
                        },
                    )
        else:  # raster
            if not class_ids:
                checks["attr_values_match"] = _skipped("no class_ids declared")
            else:
                declared = set(class_ids.values())
                with rasterio.open(label) as src:
                    h = max(1, min(SAMPLE_MAX_DIM, src.height))
                    w = max(1, min(SAMPLE_MAX_DIM, src.width))
                    data = src.read(
                        1, out_shape=(h, w), resampling=Resampling.nearest
                    )
                    nodata = src.nodata
                present = set(np.unique(data).tolist())
                if nodata is not None:
                    present.discard(nodata)
                extra = sorted(present - declared, key=str)
                missing = sorted(declared - present, key=str)
                # Missing declared classes in a sample is soft info; only fail
                # on values present that aren't declared (corrupt mapping risk).
                checks["attr_values_match"] = _check(
                    len(extra) == 0,
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

    # --- 9. nonempty_after_filter ---
    try:
        if label_type == "vector":
            if attr_field is None or attr_values is None:
                checks["nonempty_after_filter"] = _skipped(
                    "no attr_field/attr_values filter"
                )
            else:
                assert label_gdf is not None
                resolved = _resolve_attr_field(label_gdf.columns, attr_field)
                if resolved is None:
                    checks["nonempty_after_filter"] = _check(
                        False, {"empty_after_filter": True, "n": 0}
                    )
                else:
                    declared_n = _normalize_values(attr_values)
                    mask = label_gdf[resolved].apply(
                        lambda v: v in declared_n
                        or str(v) in declared_n
                        or (_can_int(v) and int(v) in declared_n)
                    )
                    n = int(mask.sum())
                    checks["nonempty_after_filter"] = _check(
                        n > 0, {"empty_after_filter": n == 0, "n": n}
                    )
        else:  # raster
            if not class_ids:
                checks["nonempty_after_filter"] = _skipped("no class_ids declared")
            else:
                # Exclude background (0) if present in mapping
                fg = {v for v in class_ids.values() if v != 0}
                if not fg:
                    fg = set(class_ids.values())
                with rasterio.open(label) as src:
                    h = max(1, min(SAMPLE_MAX_DIM, src.height))
                    w = max(1, min(SAMPLE_MAX_DIM, src.width))
                    data = src.read(
                        1, out_shape=(h, w), resampling=Resampling.nearest
                    )
                n = int(np.isin(data, list(fg)).sum())
                checks["nonempty_after_filter"] = _check(
                    n > 0, {"empty_after_filter": n == 0, "n_labeled_sample": n}
                )
    except Exception as e:
        errors.append(f"nonempty_after_filter: {e}")
        checks["nonempty_after_filter"] = _check(False, str(e))


def _can_int(v: Any) -> bool:
    try:
        if isinstance(v, bool):
            return False
        int(v)
        return True
    except (TypeError, ValueError):
        return False


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
        if isinstance(detail, (dict, list)):
            row[f"check_{name}_detail"] = json.dumps(detail, default=str)
        else:
            row[f"check_{name}_detail"] = detail
    return row


def verify_dataset(
    input_dict: list[dict],
    output_report_path: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Verify all pairs in a Tiler-style input_dict; optionally write CSV."""
    results: list[VerificationResult] = []
    for item in tqdm(input_dict, desc="Verifying pairs"):
        meta = item.get("metadata") or {}
        pair_id = meta.get("id") or meta.get("aoi_id") or Path(str(item["image"])).stem
        results.append(
            verify_pair(
                image=item["image"],
                label=item.get("label"),
                pair_id=str(pair_id),
                **kwargs,
            )
        )

    rows = [_flatten_result(r) for r in results]
    df = pd.DataFrame(rows)
    # Keep structured checks accessible
    df["checks"] = [r.checks for r in results]

    if output_report_path:
        out = Path(output_report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        csv_df = df.drop(columns=["checks"])
        csv_df.to_csv(out, index=False)
        logger.info("Wrote verification report to %s", out)

    n_ok = int((df["status"] == "ok").sum()) if len(df) else 0
    n_warn = int((df["status"] == "warning").sum()) if len(df) else 0
    n_err = int((df["status"] == "error").sum()) if len(df) else 0
    logger.info(
        "Verification complete: %d ok, %d warning, %d error (total %d)",
        n_ok,
        n_warn,
        n_err,
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
        required = {"image"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must include columns {required}, got {list(df.columns)}")
        rows = []
        for _, row in df.iterrows():
            meta = {}
            if "collection" in df.columns and pd.notna(row.get("collection")):
                meta["collection"] = row["collection"]
            if "gsd" in df.columns and pd.notna(row.get("gsd")):
                meta["gsd"] = row["gsd"]
            if "id" in df.columns and pd.notna(row.get("id")):
                meta["id"] = row["id"]
            label = row.get("label")
            if pd.isna(label):
                label = None
            rows.append(
                {
                    "image": row["image"],
                    "label": label,
                    "metadata": meta,
                }
            )
        return rows

    raise ValueError(f"Unsupported input format: {p.suffix} (use .json or .csv)")


def _parse_class_ids(raw: str | None) -> dict | None:
    if raw is None:
        return None
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--class_ids must be a dict literal")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight verify image/label pairs before tiling."
    )
    parser.add_argument(
        "input",
        help="Path to JSON list or CSV with image[, label] columns",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="verification_report.csv",
        help="Output CSV report path",
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
        default=["red", "green", "blue"],
        help="STAC common-name bands (default: red green blue)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    attr_values = None
    if args.attr_values is not None:
        attr_values = []
        for v in args.attr_values:
            try:
                attr_values.append(int(v))
            except ValueError:
                attr_values.append(v)

    attr_field: Union[str, list[str], None] = args.attr_field
    if attr_field is not None and len(attr_field) == 1:
        attr_field = attr_field[0]

    input_dict = _load_input_dict(args.input)
    df = verify_dataset(
        input_dict,
        output_report_path=args.output,
        attr_field=attr_field,
        attr_values=attr_values,
        class_ids=_parse_class_ids(args.class_ids),
        bands_expected=args.bands_expected,
        bands_requested=args.bands_requested,
    )
    n_err = int((df["status"] == "error").sum()) if len(df) else 0
    return 1 if n_err else 0


if __name__ == "__main__":
    # --- scratch: geoeye-1 FOM csv (delete before shipping) ---
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    csv_path = Path(
        "/home/valhassa/Projects/notebooks/data_prep/notebooks/fom/csv/"
        "processed/geoeye-1.csv"
    )
    raw = pd.read_csv(csv_path)
    # quick subset; drop `.head(...)` to run the full sheet
    raw = raw.head(3)

    input_dict = [
        {
            "image": row["image_url"],
            "label": row["label_path"] if pd.notna(row["label_path"]) else None,
            "metadata": {
                "id": row["image_name"],
                "collection": row.get("collection"),
                "gsd": row.get("gsd"),
                "split": row.get("split"),
            },
        }
        for _, row in raw.iterrows()
    ]

    report = verify_dataset(
        input_dict,
        output_report_path="/tmp/geoeye-1_verify_report.csv",
        attr_field=["Quatreclasses", "class"],
        attr_values=[1, 2, 3, 4],
        class_ids={
            "background": 0,
            "fore": 1,
            "hydro": 2,
            "road": 3,
            "building": 4,
        },
        bands_expected=4,
        bands_requested=["red", "green", "blue", "nir"],
    )
    print(report[["id", "status", "errors"]].to_string(index=False))
    print("wrote /tmp/geoeye-1_verify_report.csv")
    raise SystemExit(1 if (report["status"] == "error").any() else 0)
