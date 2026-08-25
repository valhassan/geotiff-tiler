# GeoTIFF Tiler

Cut training patches from image/label pairs. You pass a list of dicts — no input files.

```python
pairs = [{
    "image": "./path/to/image.tif",
    "label": "./path/to/label.gpkg",
    "metadata": {
        "split": "trn",            # trn or tst
        "collection": "worldview-3",
        "gsd": 0.46,               # any extra keys you want
    },
}]
```

## Pipeline

**val_plan** runs once on *all sensors* (trn labeled pairs only). **verify** and **tile** run per sensor.

```python
from geotiff_tiler.split_planner import run_planner
from geotiff_tiler.verify import verify_dataset
from geotiff_tiler.tiler import Tiler

run_planner(
    all_pairs,                    # every sensor, trn+tst ok (tst ignored)
    "split_plan.json",
    class_ids={"background": 0, "fore": 1},
    attr_fields=["class"],
    attr_values=[1],
)

verify_dataset(wv3_pairs, output_report_path="report.csv")
Tiler(
    input_dict=wv3_pairs,
    patch_size=(256, 256),
    stride=128,
    output_dir="./out",
    split_plan="split_plan.json",
    prefix="worldview-3",
).create_tiles()
```

`python -m geotiff_tiler` prints this usage. Dependencies are in `requirements.txt`.

## License

MIT. Victor Alhassan (victor.alhassan@nrcan-rncan.gc.ca)
