"""Synthetic tests for split_planner (run as a script)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import unary_union

from geotiff_tiler.split_planner import (
    classify_patch,
    merge_adjacent_rects,
    run_planner,
    val_rects_for_image,
)

CLASS_IDS = {"background": 0, "fore": 1, "hydro": 2, "road": 3, "building": 4}
PATCH, STRIDE = 512, 256


def _make_scene(out: Path, name, sensor, size_px, gsd, origin, seed):
    rng = np.random.default_rng(seed)
    w = h = size_px
    tr = from_origin(origin[0], origin[1], gsd, gsd)
    img_path = out / f"{name}.tif"
    with rasterio.open(
        img_path,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=1,
        dtype="uint8",
        crs="EPSG:32618",
        transform=tr,
    ) as dst:
        dst.write(rng.integers(0, 255, (1, h, w), dtype=np.uint8))

    feats, vals = [], []
    ext = size_px * gsd
    for _ in range(6):
        cx, cy = rng.uniform(0.1, 0.9, 2) * ext
        s = rng.uniform(0.08, 0.2) * ext
        feats.append(
            box(
                origin[0] + cx,
                origin[1] - cy - s,
                origin[0] + cx + s,
                origin[1] - cy,
            )
        )
        vals.append(1)
    cx, cy = rng.uniform(0.2, 0.6, 2) * ext
    s = 0.15 * ext
    feats.append(
        box(
            origin[0] + cx,
            origin[1] - cy - s,
            origin[0] + cx + s,
            origin[1] - cy,
        )
    )
    vals.append(2)
    for _ in range(3):
        y = rng.uniform(0.1, 0.9) * ext
        feats.append(
            box(origin[0], origin[1] - y - 8 * gsd, origin[0] + ext, origin[1] - y)
        )
        vals.append(3)
    for _ in range(25):
        cx = rng.uniform(0.65, 0.95) * ext
        cy = rng.uniform(0.65, 0.95) * ext
        s = rng.uniform(15, 30) * gsd
        feats.append(
            box(
                origin[0] + cx,
                origin[1] - cy - s,
                origin[0] + cx + s,
                origin[1] - cy,
            )
        )
        vals.append(4)
    gdf = gpd.GeoDataFrame({"class": vals}, geometry=feats, crs="EPSG:32618")
    lbl_path = out / f"{name}.gpkg"
    gdf.to_file(lbl_path, driver="GPKG")
    return {
        "image": str(img_path),
        "label": str(lbl_path),
        "metadata": {"split": "trn", "collection": sensor},
    }


def test_classify_and_merge():
    a = [0.0, 0.0, 10.0, 10.0]
    b = [10.0, 0.0, 20.0, 10.0]
    merged = merge_adjacent_rects([a, b])
    assert len(merged) == 1
    assert merged[0] == [0.0, 0.0, 20.0, 10.0]

    rects = merged
    tol = 0.1
    assert classify_patch((1, 1, 4, 4), rects, tol) == "val"
    # spanning the original seam is inside the union
    assert classify_patch((8, 1, 12, 4), rects, tol) == "val"
    assert classify_patch((30, 30, 40, 40), rects, tol) == "trn"
    assert classify_patch((-2, 1, 2, 4), rects, tol) is None

    # boundless window clipped to image is fully inside → val
    img_bounds = (0.0, 0.0, 20.0, 10.0)
    assert (
        classify_patch((-1, 1, 4, 4), rects, tol, img_bounds=img_bounds) == "val"
    )


def test_planner_e2e():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        pairs = [
            _make_scene(out, "wv3_a", "worldview-3", 3072, 0.31, (500000, 5000000), 1),
            _make_scene(out, "wv3_b", "worldview-3", 4096, 0.31, (510000, 5000000), 2),
            _make_scene(
                out, "ps2_a", "planetscope-2", 3072, 3.0, (520000, 5100000), 3
            ),
        ]
        plan_file = out / "split_plan.json"
        plan = run_planner(
            pairs,
            plan_file,
            CLASS_IDS,
            ["class"],
            [1, 2, 3, 4],
            patch_size=PATCH,
            stride=STRIDE,
            val_ratio=0.2,
        )

        for s, d in plan["sensors"].items():
            r = d["est_val_patches"] / d["total_patches"]
            print(f"sensor {s}: val ratio {r:.3f}")
            assert 0.10 <= r <= 0.35, f"sensor {s} ratio {r} out of range"

        t, v = plan["target_distribution"], plan["global"]["val_distribution"]
        gap = sum(abs(t[c] - v[c]) for c in t)
        print(f"L1 gap(target, val) = {gap:.4f}")
        assert gap < 0.30
        for c in ["fore", "hydro", "road", "building"]:
            assert v[c] > 0, f"class {c} absent from val"

        rec = plan["images"]["wv3_a"]
        rects = val_rects_for_image(
            plan, "wv3_a", rasterio.crs.CRS.from_string(rec["crs"])
        )
        assert rects, "no val rects for wv3_a"
        gsd = 0.31
        union = unary_union([box(*r) for r in rects])
        r = rects[0]
        inside = (
            r[0] + gsd,
            r[1] + gsd,
            r[0] + gsd + PATCH * gsd * 0.5,
            r[1] + gsd + PATCH * gsd * 0.5,
        )
        outside = (
            union.bounds[2] + 1000,
            union.bounds[3] + 1000,
            union.bounds[2] + 1100,
            union.bounds[3] + 1100,
        )
        ex = union.bounds[0]
        straddle = (ex - 50, r[1] + gsd, ex + 50, r[1] + gsd + 100)
        assert classify_patch(inside, rects, gsd) == "val"
        assert classify_patch(outside, rects, gsd) == "trn"
        assert classify_patch(straddle, rects, gsd) is None
        print("leakage classification ok")

        before = {k: json.dumps(v, sort_keys=True) for k, v in plan["images"].items()}
        pairs2 = pairs + [
            _make_scene(out, "ps2_b", "planetscope-2", 3072, 3.0, (530000, 5100000), 4)
        ]
        plan2 = run_planner(
            pairs2,
            plan_file,
            CLASS_IDS,
            ["class"],
            [1, 2, 3, 4],
            patch_size=PATCH,
            stride=STRIDE,
            val_ratio=0.2,
        )
        for k, v in before.items():
            assert json.dumps(plan2["images"][k], sort_keys=True) == v, (
                f"assignment for {k} changed on rerun"
            )
        assert plan2["images"]["ps2_b"]["status"] == "assigned"
        r2 = (
            plan2["sensors"]["planetscope-2"]["est_val_patches"]
            / plan2["sensors"]["planetscope-2"]["total_patches"]
        )
        print(f"stickiness ok; planetscope-2 ratio after increment: {r2:.3f}")
        assert 0.10 <= r2 <= 0.35

        plan3 = run_planner(
            pairs2,
            plan_file,
            CLASS_IDS,
            ["class"],
            [1, 2, 3, 4],
            patch_size=PATCH,
            stride=STRIDE,
            val_ratio=0.2,
        )
        assert {k: json.dumps(v, sort_keys=True) for k, v in plan3["images"].items()} == {
            k: json.dumps(v, sort_keys=True) for k, v in plan2["images"].items()
        }
        print("no-op rerun stable")


def main() -> int:
    test_classify_and_merge()
    print("classify/merge ok")
    test_planner_e2e()
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
