"""CLI: python -m geotiff_tiler <verify|val_plan|tile> [args]."""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)

_CLASS_IDS_HELP = "Dict literal, e.g. \"{'background': 0, 'fore': 1}\""


def _add_label_args(
    parser: argparse.ArgumentParser, *, class_ids_required: bool = False
) -> None:
    parser.add_argument("--attr_field", nargs="+", default=None)
    parser.add_argument("--attr_values", nargs="+", default=None)
    parser.add_argument(
        "--class_ids",
        required=class_ids_required,
        default=None,
        help=_CLASS_IDS_HELP,
    )


def _add_patch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--label_threshold", type=float, default=0.01)


def _scalar_attr_field(attr_field: list[str] | None) -> str | list[str] | None:
    if attr_field is not None and len(attr_field) == 1:
        return attr_field[0]
    return attr_field


def _cmd_verify(args: argparse.Namespace) -> int:
    from geotiff_tiler.lib.pairs import load_pairs, parse_attr_values, parse_class_ids
    from geotiff_tiler.verify import verify_dataset

    df = verify_dataset(
        load_pairs(args.input),
        output_report_path=args.output,
        attr_field=_scalar_attr_field(args.attr_field),
        attr_values=parse_attr_values(args.attr_values),
        class_ids=parse_class_ids(args.class_ids),
        bands_expected=args.bands_expected,
        bands_requested=args.bands_requested,
    )
    n_err = int((df["status"] == "error").sum()) if len(df) else 0
    return 1 if n_err else 0


def _cmd_val_plan(args: argparse.Namespace) -> int:
    from geotiff_tiler.lib.pairs import parse_attr_values, parse_class_ids
    from geotiff_tiler.split_planner import _planner_pairs, run_planner

    pairs = _planner_pairs(args.input_files)
    logger.info(
        "Planning split for %d trn images from %d files",
        len(pairs),
        len(args.input_files),
    )
    run_planner(
        pairs,
        args.plan_file,
        parse_class_ids(args.class_ids, required=True),
        args.attr_field,
        parse_attr_values(args.attr_values),
        args.patch_size,
        args.stride,
        args.val_ratio,
        args.cell_strides,
        args.coarse_factor,
        args.label_threshold,
    )
    return 0


def _cmd_tile(args: argparse.Namespace) -> int:
    from geotiff_tiler.lib.pairs import load_pairs, parse_attr_values, parse_class_ids
    from geotiff_tiler.tiler import Tiler

    tiler = Tiler(
        input_dict=load_pairs(args.input_file),
        patch_size=(args.patch_size, args.patch_size),
        bands_requested=args.bands_requested,
        band_indices=args.band_indices,
        stride=args.stride,
        attr_field=_scalar_attr_field(args.attr_field),
        attr_values=parse_attr_values(args.attr_values),
        class_ids=parse_class_ids(args.class_ids),
        discard_empty=not args.keep_empty,
        label_threshold=args.label_threshold,
        split=args.split,
        prefix=args.prefix,
        output_dir=args.output_dir,
        output_format=args.output_format,
        split_plan=args.split_plan,
        apply_dra=args.apply_dra,
        dra_cal=args.dra_cal,
        erosion_classes=args.erosion_classes,
        target_gap_m=args.target_gap_m,
        building_class_val=args.building_class_val,
        road_class_val=args.road_class_val,
    )
    result = tiler.create_tiles()
    if result["failed"] > 0 and args.retry > 0:
        tiler.retry_failed_images(max_retries=args.retry)
    return 1 if result["failed"] else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m geotiff_tiler",
        description="Pre-flight, split, then cut patches from image/label pairs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    verify = sub.add_parser("verify", help="pre-flight checks on image/label pairs")
    verify.add_argument("input", help="JSON list or CSV with image/image_url")
    verify.add_argument(
        "-o", "--output", default="verification_report.csv", help="Output CSV path"
    )
    _add_label_args(verify)
    verify.add_argument("--bands_expected", type=int, default=None)
    verify.add_argument(
        "--bands_requested",
        nargs="+",
        default=None,
        help="STAC common-name bands (default: validate_image default)",
    )
    verify.set_defaults(_run=_cmd_verify)

    val_plan = sub.add_parser(
        "val_plan", help="build split_plan.json (all sensors, before any tiling)"
    )
    val_plan.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="CSV (image/image_url, label/label_path, collection) or JSON pairs",
    )
    val_plan.add_argument("--plan_file", required=True)
    _add_patch_args(val_plan)
    val_plan.add_argument("--val_ratio", type=float, default=0.2)
    val_plan.add_argument("--cell_strides", type=int, default=4)
    val_plan.add_argument("--coarse_factor", type=int, default=4)
    _add_label_args(val_plan, class_ids_required=True)
    val_plan.set_defaults(_run=_cmd_val_plan)

    tile = sub.add_parser("tile", aliases=["patch"], help="cut patches")
    tile.add_argument("--input_file", "--input", required=True)
    tile.add_argument("--output_dir", "-o", required=True)
    tile.add_argument("--split_plan", default=None)
    _add_patch_args(tile)
    tile.add_argument(
        "--bands_requested", nargs="+", default=["red", "green", "blue", "nir"]
    )
    tile.add_argument("--band_indices", type=int, nargs="+", default=None)
    _add_label_args(tile)
    tile.add_argument("--keep_empty", action="store_true")
    tile.add_argument("--split", choices=("trn", "tst"), default="trn")
    tile.add_argument("--prefix", default="satellite")
    tile.add_argument("--output_format", choices=("tar", "csv"), default="tar")
    tile.add_argument("--retry", type=int, default=3)
    tile.add_argument("--apply_dra", action="store_true")
    tile.add_argument("--dra_cal", default=None)
    tile.add_argument("--erosion_classes", nargs="+", default=None)
    tile.add_argument("--target_gap_m", type=float, default=None)
    tile.add_argument("--building_class_val", type=int, default=None)
    tile.add_argument("--road_class_val", type=int, default=None)
    tile.set_defaults(_run=_cmd_tile)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("rasterio").setLevel(logging.ERROR)
    return int(args._run(args) or 0)
