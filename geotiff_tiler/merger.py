import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import webdataset as wds
from tqdm import tqdm

from geotiff_tiler.config import logging_config  # noqa: F401
from geotiff_tiler.tiling_manifest import TilingManifest

logger = logging.getLogger(__name__)


class GeospatialDatasetMerger:
    def __init__(
        self,
        base_dir: str,
        output_dir: str,
        clahe_dir: str = None,
        prefix: str = "dataset",
        max_shard_size: float = 2e9,
    ):
        self.base_dir = Path(base_dir)
        self.clahe_dir = Path(clahe_dir) if clahe_dir else None
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.max_shard_size = max_shard_size

        self.prefix_dir = self.output_dir / self.prefix
        self.prefix_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_manager = TilingManifest(str(self.output_dir), self.prefix)
        self.manifest_manager.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        self.base_source_manifest = self._find_manifest(self.base_dir)
        self.total_base_patches = self._get_total_patches(self.base_source_manifest)

        # Safely handle the presence or absence of a CLAHE/Variant directory
        if self.clahe_dir:
            self.clahe_source_manifest = self._find_manifest(self.clahe_dir)
            self.total_clahe_patches = self._get_total_patches(
                self.clahe_source_manifest
            )
        else:
            self.clahe_source_manifest = {}
            self.total_clahe_patches = 0
            logger.info(
                "No variant directory provided. Operating in Single-Sensor Mode."
            )

        # State & Routing Trackers
        self.base_split_truth = {}
        self.val_decision_map = {}
        self.available_clahe_patches = set()
        self.initialized_images = set()

        # Validation Balance Trackers
        self.val_pixels_base = defaultdict(int)
        self.val_pixels_clahe = defaultdict(int)

        # Math & I/O Trackers
        self.global_patch_counts = {"trn": 0, "val": 0, "tst": 0}
        self.global_class_pixels = defaultdict(int)
        self.active_writers = {}
        self.current_shard_indices = {"trn": 0, "val": 0, "tst": 0}
        self.current_shard_sizes = {"trn": 0, "val": 0, "tst": 0}

    def _find_manifest(self, dir_path: Path) -> dict:
        manifests = list(dir_path.glob("*_manifest.json"))
        if manifests:
            with open(manifests[0], "r") as f:
                return json.load(f)
        logger.error(f"Could not find a *_manifest.json in {dir_path}")
        return {}

    def _get_total_patches(self, manifest_data: dict) -> int:
        try:
            counts = manifest_data.get("statistics", {}).get("patch_counts", {})
            return sum(counts.values()) if counts else None
        except Exception:
            return None

    def _clean_image_name(self, img_name: str) -> str:
        return re.sub(
            r"_(red-green-blue-nir|clahe.*|base.*)", "", img_name, flags=re.IGNORECASE
        )

    def _get_patch_id(self, meta: dict) -> str:
        pm = meta.get("patch_metadata", {})
        img_name = pm.get("image_name", "")
        base_img_name = self._clean_image_name(img_name)
        x, y = pm["pixel_coordinates"]
        return f"{base_img_name}_{x}_{y}"

    def _estimate_patch_size(self, img: np.ndarray, lbl: np.ndarray, meta: dict) -> int:
        return img.nbytes + lbl.nbytes + len(json.dumps(meta).encode("utf-8"))

    def _initialize_image(self, image_name: str, meta: dict):
        if image_name in self.initialized_images:
            return

        clahe_images = self.clahe_source_manifest.get("processed_images", {})
        base_images = self.base_source_manifest.get("processed_images", {})
        cleaned_name = self._clean_image_name(image_name)

        source_data = (
            clahe_images.get(image_name)
            or clahe_images.get(cleaned_name)
            or base_images.get(image_name)
            or base_images.get(cleaned_name)
        )

        if source_data:
            self.manifest_manager.update_image_metadata(
                image_name,
                {
                    "status": "completed",
                    "path": source_data.get("path", ""),
                    "label_path": source_data.get("label_path", ""),
                    "metadata": source_data.get(
                        "metadata", meta.get("patch_metadata", {})
                    ),
                    "sensor_type": source_data.get("sensor_type", "unknown"),
                },
            )
            self.manifest_manager.mark_image_in_progress(image_name)
        self.initialized_images.add(image_name)

    # ---------------------------------------------------------
    # EXPLICIT SHARD MANAGEMENT
    # ---------------------------------------------------------
    def _get_or_create_writer(self, split: str, img_name: str):
        if split not in self.active_writers:
            idx = self.current_shard_indices[split]
            path = self.prefix_dir / split / f"{self.prefix}-{split}-{idx:06d}.tar"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.active_writers[split] = wds.TarWriter(str(path))

            self.manifest_manager.update_shard_record(
                prefix=self.prefix,
                split=split,
                shard_index=idx,
                size_bytes=0,
                patch_count=self.global_patch_counts[split],
                status="OPEN",
                images=[img_name],
            )
        return self.active_writers[split]

    def _rotate_writer(self, split: str):
        if split in self.active_writers:
            idx = self.current_shard_indices[split]
            self.active_writers[split].close()

            path = self.prefix_dir / split / f"{self.prefix}-{split}-{idx:06d}.tar"
            final_size = os.path.getsize(path) if path.exists() else 0

            self.manifest_manager.update_shard_record(
                prefix=self.prefix,
                split=split,
                shard_index=idx,
                size_bytes=final_size,
                patch_count=self.global_patch_counts[split],
                status="CLOSED",
            )
            self.manifest_manager.close_shard(self.prefix, split, idx)
            del self.active_writers[split]
            self.current_shard_sizes[split] = 0
            self.current_shard_indices[split] += 1

    def _write_and_track(
        self, split: str, tar_key: str, img: np.ndarray, lbl: np.ndarray, meta: dict
    ):
        pm = meta["patch_metadata"]
        img_name = pm["image_name"]
        self._initialize_image(img_name, meta)

        estimated_size = self._estimate_patch_size(img, lbl, meta)
        if self.current_shard_sizes[split] + estimated_size > self.max_shard_size:
            self._rotate_writer(split)

        writer = self._get_or_create_writer(split, img_name)
        writer.write(
            {
                "__key__": tar_key,
                "image_patch.npy": img,
                "label_patch.npy": lbl,
                "metadata.json": meta,
            }
        )

        self.current_shard_sizes[split] += estimated_size
        self.global_patch_counts[split] += 1
        idx = self.current_shard_indices[split]
        x, y = pm["pixel_coordinates"]

        self.manifest_manager.update_shard_record(
            prefix=self.prefix,
            split=split,
            shard_index=idx,
            size_bytes=self.current_shard_sizes[split],
            patch_count=self.global_patch_counts[split],
            status="OPEN",
            images=[img_name],
        )
        self.manifest_manager.update_image_patch_info(img_name, split, idx)
        self.manifest_manager.mark_patch_completed(img_name, x, y)

        if split == "trn":
            self.manifest_manager.update_running_statistics(self.prefix, img)

        bincount = np.bincount(lbl.flatten())
        for cls, count in enumerate(bincount):
            self.global_class_pixels[cls] += count

    # ---------------------------------------------------------
    # MAIN EXECUTION PHASES
    # ---------------------------------------------------------
    def _get_safe_tars(self, dir_path: Path):
        out_resolved = self.output_dir.resolve()
        tars = [
            f
            for f in dir_path.rglob("*.tar")
            if out_resolved not in f.resolve().parents and f.resolve() != out_resolved
        ]
        return sorted(tars)

    def phase_1_build_anchor(self):
        logger.info("Phase 1: Building Split Truth Anchor from Base Dataset...")
        base_tars = self._get_safe_tars(self.base_dir)
        dataset = (
            wds.WebDataset([str(f) for f in base_tars], shardshuffle=False)
            .decode()
            .to_tuple("metadata.json")
        )
        for (meta,) in tqdm(
            dataset,
            total=self.total_base_patches,
            desc="Phase 1: Anchoring Base Splits",
        ):
            self.base_split_truth[self._get_patch_id(meta)] = meta["patch_metadata"][
                "split"
            ]

    def execute_shuffle(self):
        """Mode: Single-Sensor Scrambler. Randomizes patches across the physical disk."""
        logger.info("Executing Deep Shuffled Writes for Single Sensor...")

        source_tars = self._get_safe_tars(self.base_dir)
        stream = (
            wds.WebDataset(
                [str(f) for f in source_tars],
                shardshuffle=True,
            )
            .decode()
            .to_tuple("__key__", "image_patch.npy", "label_patch.npy", "metadata.json")
        )

        shuffled_ds = wds.DataPipeline(stream, wds.shuffle(10000, initial=2500))

        for key, img, lbl, meta in tqdm(
            shuffled_ds, total=self.total_base_patches, desc="Writing Shuffled Shards"
        ):
            split = meta["patch_metadata"]["split"]
            self._write_and_track(split, key, img, lbl, meta)

        for split in list(self.active_writers.keys()):
            self._rotate_writer(split)

        self._finalize_manifest()
        self.export_normalization_stats()

    def execute_merge(self):
        # ---------------------------------------------------------
        # PHASE 2: MAP DECISIONS (NO I/O WRITES)
        # ---------------------------------------------------------
        logger.info("Phase 2: Mapping Validation Decisions (Fast I/O Mode)...")
        clahe_tars = self._get_safe_tars(self.clahe_dir)

        # Decoding ONLY label and metadata to massively speed up mapping
        clahe_ds = (
            wds.WebDataset([str(f) for f in clahe_tars], shardshuffle=False)
            .decode()
            .to_tuple("label_patch.npy", "metadata.json")
        )

        skipped_clahe = 0
        example_miss = None

        for lbl, meta in tqdm(
            clahe_ds,
            total=self.total_clahe_patches,
            desc="Phase 2: Mapping Route Decisions",
        ):
            patch_id = self._get_patch_id(meta)
            base_split = self.base_split_truth.get(patch_id)

            if not base_split:
                skipped_clahe += 1
                if example_miss is None:
                    example_miss = patch_id
                continue

            self.available_clahe_patches.add(patch_id)

            if base_split == "val":
                bincount = np.bincount(lbl.flatten())
                penalty_clahe = sum(
                    abs((self.val_pixels_clahe[c] + count) - self.val_pixels_base[c])
                    for c, count in enumerate(bincount)
                    if count > 0
                )
                penalty_base = sum(
                    abs(self.val_pixels_clahe[c] - (self.val_pixels_base[c] + count))
                    for c, count in enumerate(bincount)
                    if count > 0
                )

                if penalty_clahe <= penalty_base:
                    for c, count in enumerate(bincount):
                        self.val_pixels_clahe[c] += count
                    self.val_decision_map[patch_id] = "CLAHE"
                else:
                    for c, count in enumerate(bincount):
                        self.val_pixels_base[c] += count
                    self.val_decision_map[patch_id] = "BASE"

        if skipped_clahe > 0:
            logger.warning(
                f"CRITICAL: {skipped_clahe} CLAHE patches skipped. Example Miss ID: '{example_miss}'"
            )

        # ---------------------------------------------------------
        # PHASE 3: RANDOM MIX & EXECUTE WRITES
        # ---------------------------------------------------------
        logger.info("Phase 3: Fusing Streams and Executing Shuffled Writes...")

        base_tars = self._get_safe_tars(self.base_dir)

        # Open both datasets fully and inject a boolean flag to track origin
        base_stream = (
            wds.WebDataset([str(f) for f in base_tars], shardshuffle=False)
            .decode()
            .to_tuple("__key__", "image_patch.npy", "label_patch.npy", "metadata.json")
            .map(lambda s: s + (False,))
        )
        clahe_stream = (
            wds.WebDataset([str(f) for f in clahe_tars], shardshuffle=False)
            .decode()
            .to_tuple("__key__", "image_patch.npy", "label_patch.npy", "metadata.json")
            .map(lambda s: s + (True,))
        )

        # The Chaotic Blender (Wrapped securely in DataPipeline)
        mixed_ds = wds.DataPipeline(
            wds.RandomMix([base_stream, clahe_stream], longest=True),
            wds.shuffle(5000, initial=1000),
        )
        total_combined = self.total_base_patches + self.total_clahe_patches

        for key, img, lbl, meta, is_clahe in tqdm(
            mixed_ds, total=total_combined, desc="Phase 3: Writing Shards"
        ):
            patch_id = self._get_patch_id(meta)
            base_split = self.base_split_truth.get(patch_id)
            if not base_split:
                continue

            meta["patch_metadata"]["split"] = base_split
            has_clahe = patch_id in self.available_clahe_patches

            # ---- WRITE ROUTING LOGIC ----

            if base_split == "trn":
                if has_clahe:
                    # 3:1 Probabilistic Undersampling via Cryptographic Hash
                    hash_val = int(
                        hashlib.md5(patch_id.encode("utf-8")).hexdigest(), 16
                    )
                    deterministic_roll = (hash_val % 100) / 100.0
                    winner = "CLAHE" if deterministic_roll < 0.75 else "BASE"

                    if winner == "CLAHE" and is_clahe:
                        self._write_and_track("trn", key, img, lbl, meta)
                    elif winner == "BASE" and not is_clahe:
                        self._write_and_track("trn", key, img, lbl, meta)
                else:
                    if not is_clahe:  # Preserve Geographic Orphans
                        self._write_and_track("trn", key, img, lbl, meta)

            elif base_split == "val":
                winner = self.val_decision_map.get(patch_id, "BASE")
                if winner == "CLAHE" and is_clahe:
                    self._write_and_track("val", key, img, lbl, meta)
                elif winner == "BASE" and not is_clahe:
                    self._write_and_track("val", key, img, lbl, meta)

            elif base_split == "tst":
                if has_clahe and is_clahe:  # Test strictly prefers CLAHE
                    self._write_and_track("tst", key, img, lbl, meta)
                elif not has_clahe and not is_clahe:
                    self._write_and_track("tst", key, img, lbl, meta)

        for split in list(self.active_writers.keys()):
            self._rotate_writer(split)

        self._finalize_manifest()
        self.export_normalization_stats()

    def _finalize_manifest(self):
        logger.info("Phase 4: Finalizing mathematically accurate Manifest...")
        CLASS_MAP = {0: "background", 1: "fore", 2: "hydro", 3: "road", 4: "building"}
        total_global_pixels = sum(self.global_class_pixels.values()) or 1

        distribution = {
            CLASS_MAP.get(k, str(k)): (v / total_global_pixels)
            for k, v in self.global_class_pixels.items()
        }
        self.manifest_manager.update_class_distribution(distribution)

        for img_name in self.initialized_images:
            self.manifest_manager.mark_image_completed(img_name)

        # 1. Native Save
        self.manifest_manager.save_manifest()

        # 2. IRONCLAD POST-SAVE PATCH (Fixes the 0s)
        try:
            manifest_path = self.manifest_manager.manifest_path
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)

            manifest_data["statistics"]["patch_counts"] = self.global_patch_counts
            total_patches = sum(self.global_patch_counts.values()) or 1
            manifest_data["statistics"]["actual_split_ratio"] = {
                k: (v / total_patches) for k, v in self.global_patch_counts.items()
            }

            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to post-patch manifest statistics: {e}")

        # 3. Final Validation
        validation = self.manifest_manager.validate_manifest_consistency()
        if not validation["is_consistent"]:
            logger.warning(
                f"Manifest minor inconsistencies found: {validation['issues']}"
            )
        else:
            logger.info("Manifest Validation Passed: 100% Consistent.")

    def export_normalization_stats(self):
        logger.info("Phase 5: Exporting Normalization Statistics...")
        output_path = self.prefix_dir / f"{self.prefix}_normalization_stats.json"
        try:
            all_stats = self.manifest_manager.get_all_dataset_statistics()
            stats_with_metadata = {
                "created_at": datetime.now().isoformat(),
                "dataset_prefix": self.prefix,
                "statistics": all_stats,
            }
            with open(output_path, "w") as f:
                json.dump(stats_with_metadata, f, indent=2)
            logger.info(f"Normalization statistics successfully saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export normalization statistics: {e}")


if __name__ == "__main__":
    merger = GeospatialDatasetMerger(
        base_dir="./data/base_dataset",
        clahe_dir="./data/clahe_dataset",
        output_dir="./data/merged_dataset",
        prefix="geoeye-1-ortho-pansharp",
        max_shard_size=2e9,
    )
    merger.phase_1_build_anchor()
    merger.execute_merge()
