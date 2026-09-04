"""CSV ingest tests for the run CLI."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd

from geotiff_tiler.cli import by_collection, csv_paths, load_pairs, prepare_dataset

CSV = """\
image_url,label_path,split,collection,gsd,image_name,bbox_center
a.tif,a.gpkg,trn,geoeye-1,0.41,a,"(-66.5, 47.0)"
b.tif,b.gpkg,tst,worldview-3,0.31,b,"(-80.1, 43.2)"
c.tif,c.gpkg,trn,worldview-2,0.46,c,"(inf, inf)"
"""


def test_prepare_dataset_isolates_metadata() -> None:
    pairs = prepare_dataset(pd.read_csv(io.StringIO(CSV)))
    assert pairs[0]["metadata"]["split"] == "trn"
    assert pairs[1]["metadata"]["split"] == "tst"
    assert pairs[0]["metadata"]["collection"] == "geoeye-1"
    assert pairs[0]["metadata"]["id"] == "a"
    assert pairs[0]["metadata"]["lon"] == -66.5
    assert pairs[0]["metadata"]["lat"] == 47.0
    assert "lat" not in pairs[2]["metadata"]
    assert pairs[0]["metadata"] is not pairs[1]["metadata"]
    assert pairs[0]["metadata"]["pair_id"].startswith("a__a__")
    assert pairs[1]["metadata"]["pair_id"].startswith("b__b__")


def test_csv_paths_and_group(root: Path) -> None:
    d = root / "csvs"
    d.mkdir()
    for name, n in (("geoeye-1.csv", 2), ("worldview-3.csv", 1)):
        pd.DataFrame({
            "image_url": [f"{name}-{i}.tif" for i in range(n)],
            "label_path": [f"{name}-{i}.gpkg" for i in range(n)],
            "split": ["trn"] * n,
            "collection": [name.replace(".csv", "")] * n,
        }).to_csv(d / name, index=False)

    paths = csv_paths([str(d)])
    assert [p.name for p in paths] == ["geoeye-1.csv", "worldview-3.csv"]
    groups = by_collection(load_pairs(paths))
    assert set(groups) == {"geoeye-1", "worldview-3"}
    assert len(groups["geoeye-1"]) == 2
    geo = groups["geoeye-1"]
    assert geo[0]["metadata"]["pair_id"].startswith(
        "geoeye-1.csv-0__geoeye-1.csv-0__"
    )
    wv = by_collection(load_pairs([d / "worldview-3.csv"]))["worldview-3"]
    assert (
        wv[0]["metadata"]["pair_id"]
        == groups["worldview-3"][0]["metadata"]["pair_id"]
    )


def main() -> int:
    test_prepare_dataset_isolates_metadata()
    print("prepare_dataset ok")
    with tempfile.TemporaryDirectory() as td:
        test_csv_paths_and_group(Path(td))
    print("csv_paths/group ok")
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
