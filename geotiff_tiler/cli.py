"""python -m geotiff_tiler — CSV in, verify / plan / tile out."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from geotiff_tiler.split_planner import assign_pair_ids, run_planner
from geotiff_tiler.tiler import Tiler
from geotiff_tiler.verify import verify_dataset

logger = logging.getLogger(__name__)

REQUIRED = ("image_url", "label_path", "split", "collection")
OPTIONAL_FLOAT = (
    "gsd",
    "red_wavelength",
    "green_wavelength",
    "blue_wavelength",
    "nir_wavelength",
)
_NUM = re.compile(r"[-+]?(?:inf|nan|\d+(?:\.\d*)?(?:e[-+]?\d+)?)", re.I)


def _opt(row: pd.Series, key: str) -> Any:
    if key not in row.index:
        return None
    v = row[key]
    return None if pd.isna(v) else v


def _lonlat(v: Any) -> tuple[float, float] | None:
    if v is None:
        return None
    if isinstance(v, str):
        nums = _NUM.findall(v)
        if len(nums) < 2:
            return None
        lon, lat = float(nums[0]), float(nums[1])
    else:
        lon, lat = float(v[0]), float(v[1])
    if not (math.isfinite(lon) and math.isfinite(lat)):
        return None
    return lon, lat


def prepare_dataset(df: pd.DataFrame) -> list[dict]:
    """Map a pair CSV to Tiler ``{image, label, metadata}`` dicts."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    dataset = []
    for _, row in df.iterrows():
        meta: dict[str, Any] = {
            "split": str(row["split"]).strip().lower(),
            "collection": str(row["collection"]),
        }
        label = _opt(row, "label_path")
        if (name := _opt(row, "image_name")) is not None:
            meta["id"] = str(name)
        if (dt := _opt(row, "datetime")) is not None:
            meta["datetime"] = str(dt)
        for key in OPTIONAL_FLOAT:
            if (v := _opt(row, key)) is not None:
                meta[key] = float(v)

        lonlat = _lonlat(_opt(row, "bbox_center"))
        if lonlat is not None:
            meta["lon"], meta["lat"] = lonlat

        dataset.append({
            "image": row["image_url"],
            "label": None if label is None else str(label),
            "metadata": meta,
        })
    return assign_pair_ids(dataset)


def csv_paths(inputs: Sequence[str]) -> list[Path]:
    """Expand dirs, globs, and files into existing CSV paths."""
    out: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found = sorted(path.glob("*.csv"))
        elif any(c in raw for c in "*?["):
            found = sorted(Path(p) for p in glob.glob(raw))
        else:
            found = [path]
        if not found:
            raise FileNotFoundError(raw)
        for p in found:
            if not p.is_file():
                raise FileNotFoundError(p)
            out.append(p)
    return out


def load_pairs(paths: Sequence[Path]) -> list[dict]:
    frames = [pd.read_csv(p) for p in paths]
    return prepare_dataset(pd.concat(frames, ignore_index=True))


def by_collection(pairs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        groups[p["metadata"]["collection"]].append(p)
    return dict(groups)


def _class_ids(s: str) -> dict[str, int]:
    raw = json.loads(s)
    if not isinstance(raw, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    out = {str(k): int(v) for k, v in raw.items()}
    bad = {k: v for k, v in out.items() if v < 0 or v >= 255}
    if bad:
        raise argparse.ArgumentTypeError(f"class ids must be in 0–254, got {bad}")
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geotiff_tiler",
        description="Verify, plan val splits, and tile from pair CSVs.",
    )
    p.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="CSV file(s), dir(s), or glob(s). One file per sensor is fine.",
    )
    p.add_argument("--verify", action="store_true")
    p.add_argument("--split_planner", action="store_true")
    p.add_argument("--tiler", action="store_true")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--plan", type=Path, help="split_plan.json (default: OUTPUT_DIR)")
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument(
        "--stride",
        type=int,
        help="default: patch-size (no overlap)",
    )
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--coarse-factor", type=int, default=2)
    p.add_argument("--label-threshold", type=float, default=0.01)
    p.add_argument("--min-valid-frac", type=float, default=0.5)
    p.add_argument("--class-ids", type=_class_ids)
    p.add_argument("--attr-field", nargs="+")
    p.add_argument("--attr-values", nargs="+", type=int)
    p.add_argument("--erosion-classes", nargs="+", type=int)
    p.add_argument("--building-class-val", type=int)
    p.add_argument("--road-class-val", type=int)
    p.add_argument(
        "--bands",
        nargs="+",
        default=["red", "green", "blue", "nir"],
    )
    p.add_argument(
        "--band-indices",
        nargs="+",
        type=int,
        help="1-based source band order, e.g. 2 3 4 1 for NRGB→RGBN",
    )
    p.add_argument("--output-format", choices=("tar", "csv"), default="tar")
    p.add_argument("--apply-dra", action="store_true")
    p.add_argument("--dra-cal", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not (args.verify or args.split_planner or args.tiler):
        parser.error("need --verify, --split_planner, and/or --tiler")
    if args.tiler and args.output_dir is None:
        parser.error("--tiler requires --output-dir")
    if args.split_planner and args.class_ids is None:
        parser.error("--split_planner requires --class-ids")
    if args.apply_dra and args.dra_cal is None:
        parser.error("--apply-dra requires --dra-cal")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    out = args.output_dir or Path(".")
    out.mkdir(parents=True, exist_ok=True)
    plan = args.plan or (out / "split_plan.json")
    stride = args.stride if args.stride is not None else args.patch_size

    pairs = load_pairs(csv_paths(args.csv))
    groups = by_collection(pairs)
    logger.info(
        "loaded %d pairs across %d collection(s): %s",
        len(pairs),
        len(groups),
        ", ".join(sorted(groups)),
    )
    if args.split_planner and len(groups) < 2:
        logger.warning(
            "planner seeing %d collection(s); val quotas will not be cross-sensor",
            len(groups),
        )

    if args.verify:
        for sensor, group in groups.items():
            report = out / f"verify_{sensor}.csv"
            verify_dataset(
                group,
                output_report_path=str(report),
                attr_field=args.attr_field,
                attr_values=args.attr_values,
                class_ids=args.class_ids,
                bands_requested=args.bands,
                band_indices=args.band_indices,
            )

    if args.split_planner:
        run_planner(
            pairs,
            plan,
            class_ids=args.class_ids,
            attr_fields=args.attr_field,
            attr_values=args.attr_values,
            patch_size=args.patch_size,
            stride=stride,
            val_ratio=args.val_ratio,
            coarse_factor=args.coarse_factor,
            label_threshold=args.label_threshold,
            min_valid_frac=args.min_valid_frac,
            bands_requested=args.bands,
            band_indices=args.band_indices,
        )

    if args.tiler:
        for sensor, group in groups.items():
            Tiler(
                input_dict=group,
                patch_size=(args.patch_size, args.patch_size),
                stride=stride,
                bands_requested=args.bands,
                band_indices=args.band_indices,
                attr_field=args.attr_field,
                attr_values=args.attr_values,
                erosion_classes=args.erosion_classes,
                building_class_val=args.building_class_val,
                road_class_val=args.road_class_val,
                class_ids=args.class_ids,
                label_threshold=args.label_threshold,
                min_valid_frac=args.min_valid_frac,
                prefix=sensor,
                output_dir=str(out),
                output_format=args.output_format,
                split_plan=plan,
                apply_dra=args.apply_dra,
                dra_cal=args.dra_cal,
            ).create_tiles()
    return 0
