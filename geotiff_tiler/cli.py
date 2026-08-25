"""python -m geotiff_tiler — pass pair dicts in Python, not files."""

from __future__ import annotations

_USAGE = """\
Pass a list of {image, label, metadata} dicts in Python. No input files.

  metadata['split']       'trn' or 'tst'
  metadata['collection']  sensor name (required by the planner)

  # all sensors, once, before tiling (trn only; tst ignored)
  from geotiff_tiler.split_planner import run_planner
  run_planner(all_pairs, "split_plan.json", class_ids, attr_fields, attr_values)

  # one sensor at a time
  from geotiff_tiler.verify import verify_dataset
  from geotiff_tiler.tiler import Tiler
  verify_dataset(sensor_pairs, output_report_path="report.csv")
  Tiler(input_dict=sensor_pairs, output_dir="out",
        split_plan="split_plan.json").create_tiles()
"""


def main(argv: list[str] | None = None) -> int:
    print(_USAGE)
    return 0
