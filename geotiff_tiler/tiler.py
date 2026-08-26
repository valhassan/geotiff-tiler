"""Cut patches from image/label pairs. Last of: verify → val_plan → tile."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import IO, Any

import numpy as np
import rasterio
import webdataset as wds
from rasterio.windows import Window, bounds as window_bounds
from tqdm import tqdm

from geotiff_tiler.lib.dra import (
    apply_dra_to_file,
    load_calibration,
    resolve_sensor,
)
from geotiff_tiler.lib.geo import clip_gdf_to_window, gdf_to_geojson
from geotiff_tiler.lib.io import (
    clip_to_intersection,
    ensure_crs_match,
    log_stage,
    prepare_vector_labels,
    validate_image,
    validate_mask,
    validate_pair,
)
from geotiff_tiler.lib.manifest import TilingManifest
from geotiff_tiler.lib.viz import create_dataset_summary_visualization
from geotiff_tiler.split_planner import (
    classify_patch,
    load_plan,
    require_split,
    val_rects_for_image,
)

logger = logging.getLogger(__name__)

_MAX_SHARD = 2 * 1024 * 1024 * 1024
_SPLITS = ("trn", "val", "tst")


def _class_distribution(
    label_path, class_ids: dict[str, int] | None
) -> dict[str, float]:
    if not class_ids:
        return {}
    with rasterio.open(label_path) as src:
        arr = src.read(1).ravel()
    max_id = max(class_ids.values())
    counts = np.bincount(arr, minlength=max_id + 1)
    totals = {
        name: int(counts[cid]) if cid < len(counts) else 0
        for name, cid in class_ids.items()
    }
    n = sum(totals.values())
    if n == 0:
        return {name: 0.0 for name in class_ids}
    return {name: c / n for name, c in totals.items()}


class Tiler:
    """Clip, rasterize labels, and write overlapping patches (tar or csv)."""

    def __init__(
        self,
        input_dict: list[dict[str, Any]],
        patch_size: tuple[int, int],
        bands_requested: list[str] | None = None,
        band_indices: list[int] | None = None,
        stride: int | None = None,
        attr_field=None,
        attr_values=None,
        erosion_classes=None,
        target_gap_m: float | None = None,
        max_gsd_for_erosion: float = 1.0,
        min_erosion_area_m2: float = 4.0,
        building_class_val: int | None = None,
        road_class_val: int | None = None,
        class_ids: dict[str, int] | None = None,
        discard_empty: bool = True,
        label_threshold: float = 0.01,
        prefix: str = "sample",
        output_dir: str | None = None,
        output_format: str = "tar",
        split_plan: str | Path | None = None,
        apply_dra: bool = False,
        dra_cal: str | Path | None = None,
    ):
        if output_format not in ("tar", "csv"):
            raise ValueError(
                f"output_format must be 'tar' or 'csv', got {output_format!r}"
            )
        if output_dir is None:
            raise ValueError("output_dir is required")
        self.input_dict = input_dict
        self.patch_size = patch_size
        self.bands_requested = bands_requested or ["red", "green", "blue", "nir"]
        self.band_indices = band_indices
        self.stride = stride if stride is not None else max(patch_size)
        self.attr_field = attr_field
        self.attr_values = attr_values
        self.erosion_classes = erosion_classes
        self.target_gap_m = target_gap_m
        self.max_gsd_for_erosion = max_gsd_for_erosion
        self.min_erosion_area_m2 = min_erosion_area_m2
        self.building_class_val = building_class_val
        self.road_class_val = road_class_val
        self.class_ids = class_ids or {}
        self.discard_empty = discard_empty
        self.label_threshold = label_threshold
        self.prefix = prefix
        self.output_dir = output_dir
        self.output_format = output_format
        if split_plan is not None and not Path(split_plan).exists():
            raise FileNotFoundError(f"split_plan not found: {split_plan}")
        self.split_plan = load_plan(split_plan) if split_plan else None
        self.apply_dra = apply_dra
        self.dra_cals = None
        if apply_dra:
            if dra_cal is None:
                raise ValueError("apply_dra=True requires dra_cal path")
            self.dra_cals = load_calibration(dra_cal)
        self.manifest = TilingManifest(output_dir, self.prefix)
        self._patch_counts = {s: 0 for s in _SPLITS}
        self._shard_indices = {s: 0 for s in _SPLITS}
        self._writers: dict = {}
        self._csv_writers: dict[str, IO[str]] = {}

    def create_tiles(self) -> dict[str, int]:
        summary = {
            "total": len(self.input_dict),
            "successful": 0,
            "skipped": 0,
            "failed": 0,
        }
        tmp_root = Path(self.output_dir) / self.prefix / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        analyses = []

        logger.info("Phase 1: prepare pairs")
        for item in tqdm(self.input_dict, desc="Processing input pairs"):
            image_path, label_path = item["image"], item["label"]
            metadata = dict(item.get("metadata") or {})
            split = require_split(metadata, image_path)
            metadata["split"] = split
            image_name = Path(str(image_path)).stem
            image_tmp = tmp_root / image_name
            if image_tmp.exists():
                shutil.rmtree(image_tmp)
            image_tmp.mkdir(parents=True, exist_ok=True)

            if self.manifest.is_image_completed(image_name):
                logger.info("Skipping already completed image: %s", image_name)
                summary["skipped"] += 1
                continue
            self.manifest.mark_image_in_progress(image_name)
            metadata["image_name"] = image_name
            metadata["patch_size"] = self.patch_size
            metadata["stride"] = self.stride

            result = self._prepare_pair(
                image_path, label_path, image_tmp, metadata.get("collection")
            )
            if result["status"] != "successful":
                logger.info("Pair %s - %s", image_path, result["reason"])
                self.manifest.mark_image_failed(image_name, result["reason"])
                summary["failed"] += 1
                continue
            if result.get("dra"):
                metadata["dra"] = result["dra"]
                self.manifest.update_image_metadata(image_name, {"dra": result["dra"]})

            dist = _class_distribution(result["label_path"], self.class_ids)
            analyses.append(
                {
                    "image_path": result["image_path"],
                    "label_path": result["label_path"],
                    "targets_paths": result.get("targets_paths") or {},
                    "label_gdf": result.get("label_gdf"),
                    "metadata": metadata,
                    "image_name": image_name,
                    "class_distribution": dist,
                }
            )
            self.manifest.update_class_distribution(dist)
            summary["successful"] += 1

        self._init_writers()
        if self.split_plan is None and any(
            a["metadata"]["split"] == "trn" for a in analyses
        ):
            logger.warning("no split_plan; all trn patches -> trn")

        logger.info("Phase 2: tile")
        for analysis in tqdm(analyses, desc="Tiling"):
            name = analysis["image_name"]
            try:
                self._tile_image(analysis)
                self.manifest.mark_image_completed(name)
            except Exception as e:
                logger.error("Error tiling image %s: %s", name, e)
                self.manifest.mark_image_failed(name, str(e))
                summary["failed"] += 1
            finally:
                analysis.pop("label_gdf", None)
                if self.output_format == "tar":
                    self._close_all_writers(flush_only=True)
                self.manifest.save_manifest()

        if self.output_format == "tar":
            self._close_all_writers(flush_only=False)
        else:
            self._close_all_csv_writers()
        self.manifest.save_manifest()
        create_dataset_summary_visualization(
            self.output_dir, self.prefix, samples_per_split=5
        )
        self._export_norm_stats()
        self._log_finish(summary)
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        return summary

    def retry_failed_images(self, max_retries: int = 3) -> dict[str, int]:
        failed = dict(self.manifest.failed_images)
        if not failed:
            logger.info("No failed images to retry")
            return {"total": 0, "successful": 0, "skipped": 0, "failed": 0}
        retry = [
            item
            for item in self.input_dict
            if Path(str(item["image"])).stem in failed
        ]
        if not retry:
            logger.warning("Failed images not found in input_dict")
            return {
                "total": len(failed),
                "successful": 0,
                "skipped": 0,
                "failed": len(failed),
            }

        original, self.input_dict = self.input_dict, retry
        totals = {"total": len(retry), "successful": 0, "skipped": 0, "failed": 0}
        try:
            for attempt in range(1, max_retries + 1):
                if not self.input_dict:
                    break
                logger.info("Retry attempt %d/%d", attempt, max_retries)
                result = self.create_tiles()
                totals["successful"] += result["successful"]
                totals["skipped"] += result["skipped"]
                still = [
                    item
                    for item in self.input_dict
                    if self.manifest.is_image_failed(Path(str(item["image"])).stem)
                ]
                self.input_dict = still
                if not still:
                    break
                if attempt < max_retries:
                    time.sleep(min(2**attempt, 30))
            totals["failed"] = len(self.input_dict)
        finally:
            self.input_dict = original
        return totals

    @log_stage(stage_name="process_single_pair", log_memory=True)
    def _prepare_pair(self, image_path, label_path, tmp_dir, collection=None):
        image_name = Path(str(image_path)).stem
        try:
            image_path = validate_image(
                image_path, self.bands_requested, self.band_indices
            )
            label_path, label_type = validate_mask(label_path)
            check = validate_pair(image_path, label_path, label_type)
            if not check["valid"]:
                return {"status": "skipped", "reason": check["reason"]}
            if check.get("special_case"):
                return {
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "status": "successful",
                    "reason": check["reason"],
                }

            image_path, label_path = ensure_crs_match(
                image_path, label_path, label_type, tmp_dir
            )
            logger.info("Image: %s", image_name)
            clipped_image, clipped_label = clip_to_intersection(
                image_path, label_path, label_type, tmp_dir
            )
            if clipped_image is None and clipped_label is None:
                return {
                    "status": "skipped",
                    "reason": "No intersection between image and label",
                }

            targets_paths: dict = {}
            label_gdf = None
            if label_type == "vector":
                clipped_label, targets_paths, label_gdf = prepare_vector_labels(
                    clipped_label,
                    clipped_image,
                    tmp_dir,
                    self.attr_field,
                    self.attr_values,
                    erosion_classes=self.erosion_classes,
                    target_gap_m=self.target_gap_m,
                    max_gsd_for_erosion=self.max_gsd_for_erosion,
                    min_erosion_area_m2=self.min_erosion_area_m2,
                    building_class_val=self.building_class_val,
                    road_class_val=self.road_class_val,
                )

            dra = None
            if self.apply_dra:
                clipped_image, dra = self._apply_dra(
                    clipped_image, tmp_dir, image_name, collection
                )
            out = {
                "image_path": str(clipped_image),
                "label_path": str(clipped_label),
                "targets_paths": targets_paths,
                "label_gdf": label_gdf,
                "status": "successful",
                "reason": "Processed successfully",
            }
            if dra is not None:
                out["dra"] = dra
            return out
        except Exception as e:
            logger.error("Error processing image %s: %s", image_name, e)
            return {"status": "failed", "reason": str(e)}

    def _apply_dra(self, image_path, tmp_dir, image_name, collection):
        sensor = resolve_sensor(collection)
        if sensor is None or sensor not in self.dra_cals:
            logger.warning(
                "%s: no DRA calibration for sensor %r — skipping", image_name, sensor
            )
            return image_path, {
                "scene_id": image_name,
                "sensor": sensor,
                "contrast_status": None,
                "edr_min": None,
                "action": "skipped",
                "reason": "no calibration for sensor",
            }
        new_path, decision = apply_dra_to_file(
            image_path, self.dra_cals[sensor], scene_id=image_name, tmp_dir=tmp_dir
        )
        return new_path, asdict(decision)

    def _init_writers(self) -> None:
        self._patch_counts = {s: 0 for s in _SPLITS}
        if self.output_format == "tar":
            for split in _SPLITS:
                idx, _, count = self.manifest.get_shard_info(self.prefix, split)
                self._shard_indices[split] = idx
                self._patch_counts[split] = count
            self._writers = {}
        else:
            self._csv_writers = {}

    @log_stage(stage_name="tiling", log_memory=True)
    def _tile_image(self, analysis: dict[str, Any]) -> None:
        image_path, label_path = analysis["image_path"], analysis["label_path"]
        image_name = analysis["image_name"]
        metadata = analysis["metadata"]
        create_val_set = metadata["split"] == "trn"
        label_gdf = analysis.get("label_gdf")
        target_srcs: dict = {}
        t0 = time.time()
        try:
            with rasterio.open(image_path) as src_image, rasterio.open(
                label_path
            ) as src_label:
                w, h, n_bands = src_image.width, src_image.height, src_image.count
                if (w, h) != (src_label.width, src_label.height):
                    raise ValueError("Image and label dimensions must match")
                if self.patch_size[0] > h or self.patch_size[1] > w:
                    raise ValueError("Patch size must be smaller than image dimensions")

                metadata["image_channels"] = n_bands
                metadata["label_channels"] = src_label.count
                n_x = (w + self.stride - 1) // self.stride
                n_y = (h + self.stride - 1) // self.stride
                total = n_x * n_y

                plan_rects: list = []
                plan_tol = abs(src_image.transform.a)
                img_bnds = tuple(src_image.bounds)
                if create_val_set and self.split_plan is not None:
                    rects = val_rects_for_image(
                        self.split_plan, image_name, src_image.crs
                    )
                    if rects is None:
                        logger.warning(
                            "%s: not in split plan — all patches -> trn", image_name
                        )
                    else:
                        plan_rects = rects

                self.manifest.update_image_metadata(
                    image_name,
                    {
                        "path": image_path,
                        "label_path": label_path,
                        "metadata": metadata,
                        "sensor_type": metadata.get("collection", "unknown"),
                        "class_distribution": analysis.get("class_distribution", {}),
                    },
                )
                out_root = Path(self.output_dir) / self.prefix
                splits = ("trn", "val") if create_val_set else ("tst",)
                for s in splits:
                    (out_root / s).mkdir(parents=True, exist_ok=True)
                    if self.output_format == "csv":
                        (out_root / s / "image").mkdir(parents=True, exist_ok=True)
                        (out_root / s / "label").mkdir(parents=True, exist_ok=True)
                        (out_root / s / "vector").mkdir(parents=True, exist_ok=True)

                logger.info(
                    "Tiling %d x %d x %d  patch=%s stride=%s",
                    h, w, n_bands, self.patch_size, self.stride,
                )
                nodata = src_image.nodata
                for k, v in analysis.get("targets_paths", {}).items():
                    p = Path(v)
                    if p.exists():
                        try:
                            target_srcs[k] = rasterio.open(v)
                        except Exception as e:
                            logger.warning("could not open target %s at %s: %s", k, v, e)
                    else:
                        logger.warning("target file missing: %s → %s", k, v)

                discarded = kept = trn_n = val_n = tst_n = 0
                with tqdm(total=total, desc="Tiling patches") as pbar:
                    for y in range(0, h, self.stride):
                        for x in range(0, w, self.stride):
                            if self.manifest.is_patch_completed(image_name, x, y):
                                kept += 1
                                pbar.update(1)
                                continue
                            window = Window(
                                col_off=x,
                                row_off=y,
                                width=self.patch_size[1],
                                height=self.patch_size[0],
                            )
                            if create_val_set:
                                split = classify_patch(
                                    window_bounds(window, src_image.transform),
                                    plan_rects,
                                    plan_tol,
                                    img_bounds=img_bnds,
                                )
                                if split is None:
                                    discarded += 1
                                    pbar.update(1)
                                    continue
                            else:
                                split = "tst"

                            label_patch = src_label.read(
                                window=window, boundless=True, fill_value=0
                            )
                            if not self._keep_patch(label_patch):
                                discarded += 1
                                continue

                            fill = nodata if nodata is not None else 0
                            image_patch = src_image.read(
                                window=window, boundless=True, fill_value=fill
                            )
                            targets_patches = {
                                k: src.read(
                                    1, window=window, boundless=True, fill_value=0
                                )
                                for k, src in target_srcs.items()
                            }
                            if nodata is not None:
                                label_patch[0, np.all(image_patch == nodata, axis=0)] = 255

                            if split == "val":
                                val_n += 1
                            elif split == "trn":
                                trn_n += 1
                            else:
                                tst_n += 1

                            patch_key = f"{self.prefix}_{image_name}_{x}_{y}"
                            all_metadata = {
                                "patch_metadata": {
                                    "patch_id": patch_key,
                                    "pixel_coordinates": [x, y],
                                    "patch_size": self.patch_size,
                                    "stride": self.stride,
                                    "split": split,
                                    "image_dtype": src_image.dtypes[0],
                                    "label_dtype": src_label.dtypes[0],
                                    "image_name": image_name,
                                    "sensor_type": metadata.get("collection", "unknown"),
                                },
                                "metadata": metadata,
                            }
                            if split == "trn":
                                try:
                                    self.manifest.update_running_statistics(
                                        self.prefix, image_patch
                                    )
                                except Exception as e:
                                    logger.error("Error updating running statistics: %s", e)

                            geojson_str = None
                            if label_gdf is not None and not label_gdf.empty:
                                try:
                                    patch_gdf = clip_gdf_to_window(
                                        label_gdf, window, src_image.transform
                                    )
                                    geojson_str = gdf_to_geojson(
                                        patch_gdf, window, src_image.transform
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "GeoJSON generation failed for %s: %s",
                                        patch_key,
                                        e,
                                    )

                            if self.output_format == "tar":
                                self._write_tar(
                                    split,
                                    patch_key,
                                    image_patch,
                                    label_patch,
                                    targets_patches,
                                    all_metadata,
                                    geojson_str,
                                    image_name,
                                    out_root / split,
                                )
                            else:
                                self._write_csv_row(
                                    split,
                                    patch_key,
                                    image_patch,
                                    label_patch,
                                    targets_patches,
                                    src_image,
                                    src_label,
                                    out_root,
                                    window,
                                    geojson_str,
                                )
                            self.manifest.mark_patch_completed(image_name, x, y)
                            self._patch_counts[split] += 1
                            kept += 1
                            if kept % 100 == 0:
                                self._checkpoint()
                            pbar.update(1)

                logger.info(
                    "%s done  trn=%d val=%d tst=%d kept=%d discarded=%d total=%d (%.1fs)",
                    image_name, trn_n, val_n, tst_n, kept, discarded, total,
                    time.time() - t0,
                )
        finally:
            for src in target_srcs.values():
                try:
                    src.close()
                except Exception:
                    pass
            image_tmp = Path(self.output_dir) / self.prefix / "tmp" / image_name
            if image_tmp.is_dir():
                shutil.rmtree(image_tmp)

    def _write_tar(
        self,
        split,
        patch_key,
        image_patch,
        label_patch,
        targets_patches,
        all_metadata,
        geojson_str,
        image_name,
        split_dir: Path,
    ) -> None:
        est = (
            image_patch.nbytes
            + label_patch.nbytes
            + len(json.dumps(all_metadata).encode())
        )
        if self._shard_size(split) + est > _MAX_SHARD:
            self._rotate_shard(split, image_name)

        writer = self._get_writer(split, split_dir)
        sample = {
            "__key__": patch_key,
            "image_patch.npy": image_patch,
            "label_patch.npy": label_patch,
            "metadata.json": all_metadata,
        }
        sample.update({f"{k}.npy": v for k, v in targets_patches.items()})
        if geojson_str is not None:
            sample["vectors.geojson"] = geojson_str.encode("utf-8")
        writer.write(sample)
        idx = self._shard_indices[split]
        self.manifest.update_shard_info(
            self.prefix,
            split,
            idx,
            self._shard_size(split),
            self._patch_counts[split] + 1,
        )
        self.manifest.update_image_patch_info(image_name, split, idx)

    def _write_csv_row(
        self,
        split,
        patch_key,
        image_patch,
        label_patch,
        targets_patches,
        src_image,
        src_label,
        output_root: Path,
        window,
        geojson_str,
    ) -> None:
        rel_img, rel_lbl, rel_targets, rel_vector = self._write_geotiff(
            image_patch,
            label_patch,
            targets_patches,
            patch_key,
            split,
            src_image,
            src_label,
            output_root,
            window,
            geojson_str,
        )
        extra = "".join(f";{p}" for p in rel_targets.values())
        vec = f";{rel_vector}" if rel_vector else ""
        self._get_csv_writer(split, output_root).write(
            f"{rel_img};{rel_lbl}{extra}{vec}\n"
        )

    def _keep_patch(self, label: np.ndarray) -> bool:
        if label.size == 0:
            return False
        nz = np.count_nonzero(label)
        if self.discard_empty and nz == 0:
            return False
        if nz / label.size < self.label_threshold:
            return False
        return True

    def _shard_path(self, base, prefix, split, idx):
        return os.path.join(base, f"{prefix}-{split}-{idx:06d}.tar")

    def _rotate_shard(self, split: str, image_name: str | None = None) -> None:
        self._close_writer(split)
        idx = self._shard_indices[split]
        self.manifest.close_shard(split, idx)
        self._shard_indices[split] = idx + 1
        images = [image_name] if image_name else []
        self.manifest.update_shard_record(
            self.prefix, split, idx + 1, 0, 0, "OPEN", images
        )

    def _get_writer(self, split, output_dir):
        if split not in self._writers:
            idx = self._shard_indices[split]
            path = self._shard_path(output_dir, self.prefix, split, idx)
            while os.path.exists(path) and os.path.getsize(path) > 0:
                logger.info("shard exists, skipping to next: %s", path)
                self._rotate_shard(split)
                idx = self._shard_indices[split]
                path = self._shard_path(output_dir, self.prefix, split, idx)
            fh = open(path, "wb")
            self._writers[split] = {
                "writer": wds.TarWriter(fh),
                "file_obj": fh,
                "path": path,
            }
        return self._writers[split]["writer"]

    def _close_writer(self, split, flush_only=False):
        info = self._writers.get(split)
        if not info:
            return
        writer, fh = info["writer"], info["file_obj"]
        if flush_only:
            if hasattr(writer, "tarfile") and hasattr(writer.tarfile, "fileobj"):
                writer.tarfile.fileobj.flush()
                os.fsync(writer.tarfile.fileobj.fileno())
            return
        writer.close()
        fh.close()
        del self._writers[split]

    def _close_all_writers(self, flush_only=False):
        for split in list(self._writers):
            self._close_writer(split, flush_only)
        if not flush_only:
            self._writers.clear()

    def _shard_size(self, split) -> int:
        info = self._writers.get(split)
        if not info:
            return 0
        writer = info["writer"]
        if hasattr(writer, "tarfile") and hasattr(writer.tarfile, "fileobj"):
            return writer.tarfile.fileobj.tell()
        if os.path.exists(info["path"]):
            return os.path.getsize(info["path"])
        return 0

    def _checkpoint(self) -> None:
        """Flush outputs + persist manifest. Same cadence for tar and csv."""
        if self.output_format == "tar":
            self._close_all_writers(flush_only=True)
        else:
            for fh in self._csv_writers.values():
                fh.flush()
                os.fsync(fh.fileno())
        self.manifest.save_manifest()

    def _get_csv_writer(self, split: str, output_dir: Path) -> IO[str]:
        if split not in self._csv_writers:
            self._csv_writers[split] = open(Path(output_dir) / f"{split}.csv", "a")
        return self._csv_writers[split]

    def _close_all_csv_writers(self) -> None:
        for fh in self._csv_writers.values():
            try:
                fh.flush()
                os.fsync(fh.fileno())
                fh.close()
            except OSError:
                pass
        self._csv_writers.clear()

    def _write_geotiff(
        self,
        image_patch,
        label_patch,
        targets_patches,
        patch_key,
        split,
        src_image,
        src_label,
        output_root: Path,
        window,
        geojson_str,
    ):
        img_dir = output_root / split / "image"
        lbl_dir = output_root / split / "label"
        img_path = img_dir / f"{patch_key}.tif"
        lbl_path = lbl_dir / f"{patch_key}_lbl.tif"
        transform = src_image.window_transform(window)
        img_profile = src_image.profile.copy()
        img_profile.update(
            width=self.patch_size[1],
            height=self.patch_size[0],
            count=image_patch.shape[0],
            driver="GTiff",
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            transform=transform,
        )
        with rasterio.open(img_path, "w", **img_profile) as dst:
            dst.write(image_patch)
        lbl_profile = src_label.profile.copy()
        lbl_profile.update(
            width=self.patch_size[1],
            height=self.patch_size[0],
            count=label_patch.shape[0],
            driver="GTiff",
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            transform=transform,
        )
        with rasterio.open(lbl_path, "w", **lbl_profile) as dst:
            dst.write(label_patch)

        rel_targets = {}
        for key, arr in targets_patches.items():
            tgt_dir = output_root / split / key
            tgt_dir.mkdir(parents=True, exist_ok=True)
            tgt_path = tgt_dir / f"{patch_key}_{key}.tif"
            profile = lbl_profile.copy()
            profile.update(count=1)
            with rasterio.open(tgt_path, "w", **profile) as dst:
                dst.write(arr[np.newaxis])
            rel_targets[key] = tgt_path.relative_to(output_root)

        rel_vector = None
        if geojson_str is not None:
            vec_dir = output_root / split / "vector"
            vec_path = vec_dir / f"{patch_key}.geojson"
            vec_path.write_text(geojson_str, encoding="utf-8")
            rel_vector = vec_path.relative_to(output_root)
        return (
            img_path.relative_to(output_root),
            lbl_path.relative_to(output_root),
            rel_targets,
            rel_vector,
        )

    def _export_norm_stats(self) -> None:
        path = (
            Path(self.output_dir) / self.prefix / f"{self.prefix}_normalization_stats.json"
        )
        try:
            stats = self.manifest.get_all_dataset_statistics()
            path.write_text(
                json.dumps(
                    {
                        "created_at": datetime.now().isoformat(),
                        "dataset_prefix": self.prefix,
                        "statistics": stats,
                    },
                    indent=2,
                )
            )
            logger.info("Normalization statistics saved to %s", path)
        except Exception as e:
            logger.error("Failed to export normalization statistics: %s", e)

    def _log_finish(self, summary: dict) -> None:
        logger.info("Processing complete: %s", summary)
        counts = self._patch_counts
        sizes = (
            self.manifest.get_total_sizes_by_split()
            if self.output_format == "tar"
            else None
        )
        extra = ""
        if sizes:
            extra = "  sizes_mb trn=%.1f val=%.1f tst=%.1f" % (
                sizes["trn"] / 1024**2,
                sizes["val"] / 1024**2,
                sizes["tst"] / 1024**2,
            )
        logger.info(
            "patches trn=%d val=%d tst=%d total=%d%s",
            counts["trn"],
            counts["val"],
            counts["tst"],
            sum(counts.values()),
            extra,
        )
        result = self.manifest.validate_manifest_consistency()
        if result["is_consistent"]:
            logger.info("Manifest validation: PASSED")
        else:
            logger.info(
                "Manifest validation: FAILED (%s)", ", ".join(result["issues"])
            )
