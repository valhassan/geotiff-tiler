import os
import json
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import webdataset as wds
from tqdm import tqdm

from geotiff_tiler.config import logging_config # noqa: F401
from geotiff_tiler.tiling_manifest import TilingManifest

logger = logging.getLogger(__name__)

class GeospatialDatasetMerger:
    def __init__(self, base_dir: str, clahe_dir: str, output_dir: str, prefix: str = "merged"):
        self.base_dir = Path(base_dir)
        self.clahe_dir = Path(clahe_dir)
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_manager = TilingManifest(str(self.output_dir), self.prefix)
        self.manifest = self.manifest_manager.manifest # Direct access for specific overrides
        
        # Load Source Manifests to inherit rich geospatial metadata
        self.base_source_manifest = self._load_json(self.base_dir / "manifest.json")
        self.clahe_source_manifest = self._load_json(self.clahe_dir / "manifest.json")
        
        # Extract total patch counts for tqdm progress bars
        self.total_base_patches = self._get_total_patches(self.base_source_manifest)
        self.total_clahe_patches = self._get_total_patches(self.clahe_source_manifest)
        
        # Trackers for Greedy Validation Routing
        self.val_pixels_base = defaultdict(int)
        self.val_pixels_clahe = defaultdict(int)
        self.val_assigned_to_clahe = set()
        self.val_assigned_to_base = set()
        
        # Trackers for Data Flow
        self.handled_clahe_patches = set() 
        self.base_split_truth = {}
        self.initialized_images = set()

        # Trackers for per-image and global stats (Due to dataset augmentation)
        self.global_patch_counts = {"trn": 0, "val": 0, "tst": 0}
        self.global_class_pixels = defaultdict(int)
        self.img_patch_counts = defaultdict(lambda: {"trn": 0, "val": 0, "tst": 0})
        self.img_class_pixels = defaultdict(lambda: defaultdict(int))
        self.img_completed_patches = defaultdict(list)

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _get_total_patches(self, manifest_data: dict) -> int:
        """Safely extracts the total patch count from a manifest for tqdm."""
        try:
            counts = manifest_data.get("statistics", {}).get("patch_counts", {})
            return sum(counts.values()) if counts else None
        except Exception:
            return None

    def _get_patch_id(self, meta: dict) -> str:
        """Generates a pure geographic identifier by stripping version suffixes."""
        pm = meta.get("patch_metadata", {})
        base_img_name = pm['image_name'].split("_clahe")[0]
        return f"{base_img_name}_{pm['pixel_coordinates'][0]}_{pm['pixel_coordinates'][1]}"

    def _initialize_image(self, image_name: str, meta: dict):
        """Deep-copies rich metadata from source manifests on first sight."""
        if image_name in self.initialized_images:
            return

        source_data = self.clahe_source_manifest.get("processed_images", {}).get(image_name)
        if not source_data:
            source_data = self.base_source_manifest.get("processed_images", {}).get(image_name.split("_clahe")[0])

        if source_data:
            self.manifest["processed_images"][image_name] = {
                "added_at": source_data.get("added_at"),
                "status": "completed",
                "path": source_data.get("path", ""),
                "label_path": source_data.get("label_path", ""),
                "metadata": source_data.get("metadata", meta.get("patch_metadata", {})),
                "sensor_type": source_data.get("sensor_type", "unknown"),
                "patches": {"total": 0, "distribution": {"trn": 0, "val": 0, "tst": 0}, "shard_locations": {"trn": [], "val": [], "tst": []}},
                "class_distribution": {}
            }
            if image_name not in self.manifest["progress"]["completed_images"]:
                self.manifest["progress"]["completed_images"].append(image_name)
                
        self.initialized_images.add(image_name)

    def _write_and_track(self, writer, split: str, tar_key: str, img: np.ndarray, lbl: np.ndarray, meta: dict):
        """Centralized writer updating WebDataset, native TilingManifest, and custom stats."""
        # Write using original WebDataset key (Prevents Base/CLAHE collision)
        writer.write({"__key__": tar_key, "image_patch.npy": img, "label_patch.npy": lbl, "metadata.json": meta})
        
        pm = meta["patch_metadata"]
        img_name = pm["image_name"]
        
        self._initialize_image(img_name, meta)
        
        # Native tracking
        self.manifest_manager.update_image_patch_info(img_name, split, writer.shard)
        if split == "trn":
            self.manifest_manager.update_running_statistics(self.prefix, img)

        # Custom stats tracking for mathematically accurate manifest
        self.global_patch_counts[split] += 1
        self.img_patch_counts[img_name][split] += 1
        
        x, y = pm["pixel_coordinates"]
        self.img_completed_patches[img_name].append(f"{x}_{y}")

        bincount = np.bincount(lbl.flatten())
        for cls, count in enumerate(bincount):
            self.global_class_pixels[cls] += count
            self.img_class_pixels[img_name][cls] += count

    def phase_1_build_anchor(self):
        logger.info("Phase 1: Building Split Truth Anchor from Base Dataset...")
        base_tars = sorted(self.base_dir.rglob("*.tar"))
        dataset = wds.WebDataset([str(f) for f in base_tars]).decode().to_tuple("metadata.json")
        
        # Wrapped with tqdm
        for (meta,) in tqdm(dataset, total=self.total_base_patches, desc="Phase 1: Anchoring Base Splits"):
            self.base_split_truth[self._get_patch_id(meta)] = meta["patch_metadata"]["split"]

    def execute_merge(self):
        writers = {
            "trn": wds.ShardWriter(str(self.output_dir / "trn" / f"{self.prefix}-trn-%06d.tar"), maxsize=2e9),
            "val": wds.ShardWriter(str(self.output_dir / "val" / f"{self.prefix}-val-%06d.tar"), maxsize=2e9),
            "tst": wds.ShardWriter(str(self.output_dir / "tst" / f"{self.prefix}-tst-%06d.tar"), maxsize=2e9),
        }
        for split in writers.keys(): (self.output_dir / split).mkdir(parents=True, exist_ok=True)

        # PHASE 2: STREAM CLAHE (Decision Phase)
        logger.info("Phase 2: Streaming CLAHE Dataset & Executing Greedy Routing...")
        clahe_tars = sorted(self.clahe_dir.rglob("*.tar"))
        clahe_ds = wds.WebDataset([str(f) for f in clahe_tars]).decode().to_tuple("__key__", "image_patch.npy", "label_patch.npy", "metadata.json")

        # Wrapped with tqdm
        for key, img, lbl, meta in tqdm(clahe_ds, total=self.total_clahe_patches, desc="Phase 2: Routing CLAHE Patches"):
            patch_id = self._get_patch_id(meta)
            base_split = self.base_split_truth.get(patch_id)
            if not base_split: continue 

            meta["patch_metadata"]["split"] = base_split

            if base_split in ["tst", "trn"]:
                self._write_and_track(writers[base_split], base_split, key, img, lbl, meta)
                self.handled_clahe_patches.add(patch_id)

            elif base_split == "val":
                bincount = np.bincount(lbl.flatten())
                penalty_clahe = sum(abs((self.val_pixels_clahe[c] + count) - self.val_pixels_base[c]) for c, count in enumerate(bincount) if count > 0)
                penalty_base = sum(abs(self.val_pixels_clahe[c] - (self.val_pixels_base[c] + count)) for c, count in enumerate(bincount) if count > 0)

                if penalty_clahe <= penalty_base:
                    for c, count in enumerate(bincount): self.val_pixels_clahe[c] += count
                    self._write_and_track(writers["val"], "val", key, img, lbl, meta)
                    self.val_assigned_to_clahe.add(patch_id)
                else:
                    for c, count in enumerate(bincount): self.val_pixels_base[c] += count
                    self.val_assigned_to_base.add(patch_id)

        logger.info(f"Greedy Router Summary -> Validation Split: {len(self.val_assigned_to_clahe)} patches (CLAHE) vs {len(self.val_assigned_to_base)} patches (Base).")

        # PHASE 3: STREAM BASE (Execution Phase)
        logger.info("Phase 3: Streaming Base Dataset for Backfill...")
        base_tars = sorted(self.base_dir.rglob("*.tar"))
        base_ds = wds.WebDataset([str(f) for f in base_tars]).decode().to_tuple("__key__", "image_patch.npy", "label_patch.npy", "metadata.json")

        # Wrapped with tqdm
        for key, img, lbl, meta in tqdm(base_ds, total=self.total_base_patches, desc="Phase 3: Backfilling Base Patches"):
            patch_id = self._get_patch_id(meta)
            split = meta["patch_metadata"]["split"]

            if split == "tst" and patch_id not in self.handled_clahe_patches:
                self._write_and_track(writers["tst"], "tst", key, img, lbl, meta)
            elif split == "trn":
                self._write_and_track(writers["trn"], "trn", key, img, lbl, meta)
            elif split == "val" and patch_id not in self.val_assigned_to_clahe:
                self._write_and_track(writers["val"], "val", key, img, lbl, meta)

        for w in writers.values(): w.close()
        
        # Finalization Steps
        self._finalize_manifest()
        self.export_normalization_stats()

    def _finalize_manifest(self):
        logger.info("Phase 4: Finalizing mathematically accurate Manifest...")
        CLASS_MAP = {0: "background", 1: "fore", 2: "hydro", 3: "road", 4: "building"}

        # Update Global Stats
        total_global_pixels = sum(self.global_class_pixels.values()) or 1
        self.manifest["statistics"]["class_distribution"] = {
            CLASS_MAP.get(k, str(k)): (v / total_global_pixels) for k, v in self.global_class_pixels.items()
        }
        self.manifest["statistics"]["patch_counts"] = self.global_patch_counts
        
        total_patches = sum(self.global_patch_counts.values()) or 1
        self.manifest["statistics"]["actual_split_ratio"] = {
            k: (v / total_patches) for k, v in self.global_patch_counts.items()
        }

        # Update Per-Image Stats
        if "completed_patches" not in self.manifest["progress"]:
            self.manifest["progress"]["completed_patches"] = {}

        for img_name in self.initialized_images:
            self.manifest["progress"]["completed_patches"][img_name] = self.img_completed_patches[img_name]
            
            img_stats = self.manifest["processed_images"][img_name]
            img_stats["patches"]["distribution"] = self.img_patch_counts[img_name]
            img_stats["patches"]["total"] = sum(self.img_patch_counts[img_name].values())
            
            total_img_pixels = sum(self.img_class_pixels[img_name].values()) or 1
            img_stats["class_distribution"] = {
                CLASS_MAP.get(k, str(k)): (v / total_img_pixels) for k, v in self.img_class_pixels[img_name].items()
            }

        # Update Shard byte sizes and status from physical disk files
        for split in ["trn", "val", "tst"]:
            for shard_entry in self.manifest["shards"].get(split, []):
                shard_path = self.output_dir / split / shard_entry["path"]
                if shard_path.exists():
                    shard_entry["size_bytes"] = os.path.getsize(shard_path)
                    shard_entry["status"] = "CLOSED"

        self.manifest_manager.save_manifest()

    def export_normalization_stats(self):
        """Export standalone normalization statistics matching native tiler output."""
        logger.info("Phase 5: Exporting Normalization Statistics...")
        output_path = self.output_dir / f"{self.prefix}_normalization_stats.json"
        
        try:
            all_stats = self.manifest_manager.get_all_dataset_statistics()
            stats_with_metadata = {
                "created_at": datetime.now().isoformat(),
                "dataset_prefix": self.prefix,
                "statistics": all_stats
            }
            
            with open(output_path, 'w') as f:
                json.dump(stats_with_metadata, f, indent=2)
            
            logger.info(f"Normalization statistics successfully saved to {output_path}")
            
            for prefix, stats in all_stats.items():
                logger.info(f"\nFinal Merged Statistics for {prefix}:")
                logger.info(f"  Bands: {stats['band_count']}")
                logger.info(f"  Patches processed: {stats['patch_count']}")
                logger.info(f"  Mean: {[f'{m:.4f}' for m in stats['mean']]}")
                logger.info(f"  Std:  {[f'{s:.4f}' for s in stats['std']]}")
                
        except Exception as e:
            logger.error(f"Failed to export normalization statistics: {e}")

# Execution Block
if __name__ == "__main__":
    merger = GeospatialDatasetMerger(
        base_dir="./data/base_dataset",
        clahe_dir="./data/clahe_dataset",
        output_dir="./data/merged_dataset",
        prefix="quickbird-2-ortho-pansharp_merged" 
    )
    merger.phase_1_build_anchor()
    merger.execute_merge()