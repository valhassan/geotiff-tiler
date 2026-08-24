# GeoTIFF Tiler

Cut training patches from image/label pairs. Run from the repo root — no install.

```bash
python -m geotiff_tiler verify pairs.csv -o report.csv
python -m geotiff_tiler val_plan --input_files a.csv b.csv \
    --plan_file split_plan.json --class_ids "{'background': 0, 'fore': 1}"
python -m geotiff_tiler tile --input_file a.csv -o ./out --split_plan split_plan.json
```

## Pipeline

1. **verify** — pre-flight checks on image/label pairs
2. **val_plan** — build `split_plan.json` across sensors, before any tiling
3. **tile** — cut patches (`patch` is an alias)

CSV inputs accept `image`/`image_url` and `label`/`label_path`; extra columns become metadata. JSON is a list of `{image, label, metadata}` dicts.

## Library

```python
from geotiff_tiler.tiler import Tiler

tiler = Tiler(
    input_dict=[{
        "image": "./path/to/image.tif",
        "label": "./path/to/label.tif",
        "metadata": {"collection": "satellite-name", "gsd": 0.5},
    }],
    patch_size=(256, 256),
    bands_requested=["red", "green", "blue", "nir"],
    stride=128,
    output_dir="./output/patches",
    prefix="dataset_v1",
)
tiler.create_tiles()
```

See `python -m geotiff_tiler <cmd> --help` for flags. Dependencies are in `requirements.txt`.

## License

MIT. Victor Alhassan (victor.alhassan@nrcan-rncan.gc.ca)
