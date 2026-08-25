# GeoTIFF Tiler

Cut training patches from image/label pairs.

CSV columns: `image_url`, `label_path`, `split`, `collection`. Optional: `gsd`,
`image_name`, `datetime`, `bbox_center`, band wavelengths.

```bash
python -m geotiff_tiler \
  --csv processed/*.csv \
  --verify --split_planner --tiler \
  --output-dir ./out \
  --class-ids '{"background":0,"fore":1}' \
  --attr-field class --attr-values 1
```

`--split_planner` sees every CSV at once. `--verify` and `--tiler` run per
`collection`. Stages can be combined or run separately; tiler reads
`OUTPUT_DIR/split_plan.json` unless `--plan` is set.

Python API is unchanged: pass a list of `{image, label, metadata}` dicts to
`run_planner` / `verify_dataset` / `Tiler`. Dependencies are in `requirements.txt`.

## License

MIT. Victor Alhassan (victor.alhassan@nrcan-rncan.gc.ca)
