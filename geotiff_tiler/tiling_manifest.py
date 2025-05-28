import os
import json
import logging
import signal
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Set, Optional, List

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy arrays and types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "_type": "ndarray",
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": obj.shape
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def numpy_decoder(dct):
    """JSON decoder that reconstructs NumPy arrays."""
    if "_type" in dct and dct["_type"] == "ndarray":
        arr = np.array(dct["data"], dtype=dct["dtype"])
        # Reshape if needed (handles empty arrays correctly)
        if "shape" in dct and arr.size > 0:
            arr = arr.reshape(dct["shape"])
        return arr
    return dct

class TilingManifest:
    """
    Dataset manifest manager for tiling operations and continuous dataset growth.
    
    Tracks processed images, patches, and shard information to enable resuming tiling
    operations and growing datasets over time. Extends the functionality of TilingCheckpoint
    by adding comprehensive dataset tracking capabilities.
    """
    
    def __init__(self, output_dir: str, prefix: str):
        """
        Initialize the manifest manager.
        
        Args:
            output_dir (str): Directory for output files and manifest
            prefix (str): Dataset prefix
        """
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.manifest_path = self.output_dir / prefix / f"{self.prefix}_manifest.json"
        
        # Track completed images and patches (from original TilingCheckpoint)
        self.completed_images: Set[str] = set()
        self.failed_images: Dict[str, str] = {}  # image_name -> reason
        self.in_progress_image: Optional[str] = None
        self.completed_patches: Dict[str, Set[str]] = {}  # image_name -> set of "x_y" strings
        
        # Track shard information (from original TilingCheckpoint)
        self.shard_indices: Dict[str, Dict[str, int]] = {}  # prefix -> {split -> index}
        self.shard_sizes: Dict[str, Dict[str, int]] = {}    # prefix -> {split -> size}
        self.patch_counts: Dict[str, Dict[str, int]] = {}   # prefix -> {split -> count}
        
        # New dataset information
        self.dataset_info = {
            "name": prefix,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "description": f"Geospatial patches for {prefix}"
        }
        
        # Enhanced shard tracking with status and additional metadata
        self.shards = {
            "trn": [],  # List of shard info dicts
            "val": [],
            "tst": []
        }
        
        # Image metadata and processing records
        self.image_metadata = {}  # image_name -> metadata dict
        
        # Dataset statistics
        self.dataset_statistics = {
            "class_distribution": {},
            "patch_counts": {"trn": 0, "val": 0, "tst": 0},
            "image_counts": {"total": 0, "completed": 0, "failed": 0, "in_progress": 0},
            "actual_split_ratio": {"trn": 0, "val": 0, "tst": 0}
        }
        self.running_statistics = {} # prefix -> band statistics
        self.stats_update_counter = 0 # Track updates for save frequency
        self.stats_save_frequency = 100 # Save every N patches
        
        # Load manifest if resuming
        if self.manifest_path.exists():
            self._load_manifest()
            logger.info(f"Resumed from manifest: {self.manifest_path}")
        
        # Register signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Set up handlers to save manifest on interrupt"""
        def handle_interrupt(signum, frame):
            logger.info("Interrupt received, saving manifest...")
            self.save_manifest()
            # Re-raise to allow normal exit processing
            raise KeyboardInterrupt("Tiling process was interrupted")
        
        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_interrupt)
        
    def _initialize_band_statistics(self, prefix: str, patch: np.ndarray):
        """Initialize statistics tracking for a new prefix/sensor configuration."""
        band_count = patch.shape[0]
        
        self.running_statistics[prefix] = {
            "band_count": band_count,
            "pixel_count": 0,
            "band_sums": np.zeros(band_count, dtype=np.float64),
            "band_sums_squared": np.zeros(band_count, dtype=np.float64),
            "dtype": str(patch.dtype),
            "first_patch_shape": patch.shape,
            "last_updated": datetime.now().isoformat(),
            "patch_count": 0  # Number of patches processed
        }
        
        logger.info(f"Initialized statistics for prefix '{prefix}' with {band_count} bands")
    
    # --- Image and Patch Tracking Methods (from original TilingCheckpoint) ---
    
    def is_image_completed(self, image_name: str) -> bool:
        """Check if an image has been fully processed"""
        return image_name in self.completed_images
    
    def is_image_failed(self, image_name: str) -> bool:
        """Check if an image has failed processing"""
        return image_name in self.failed_images
    
    def is_patch_completed(self, image_name: str, x: int, y: int) -> bool:
        """Check if a specific patch has been processed"""
        if image_name not in self.completed_patches:
            return False
        return f"{x}_{y}" in self.completed_patches[image_name]
    
    def mark_image_in_progress(self, image_name: str) -> None:
        """Mark an image as currently being processed"""
        self.in_progress_image = image_name
        # Initialize patch tracking for this image if not exists
        if image_name not in self.completed_patches:
            self.completed_patches[image_name] = set()
        
        # Update dataset statistics
        self.dataset_statistics["image_counts"]["in_progress"] = 1
        
        # Update last_updated timestamp
        self.dataset_info["last_updated"] = datetime.now().isoformat()
    
    def mark_image_completed(self, image_name: str) -> None:
        """Mark an image as completely processed"""
        self.completed_images.add(image_name)
        if image_name in self.failed_images:
            del self.failed_images[image_name]
        if self.in_progress_image == image_name:
            self.in_progress_image = None
        
        # Update image metadata
        if image_name in self.image_metadata:
            self.image_metadata[image_name]["status"] = "completed"
            self.image_metadata[image_name]["completed_at"] = datetime.now().isoformat()
        
        # Update dataset statistics
        self.update_dataset_statistics()
        
        # Save manifest after completing an image
        self.save_manifest()
    
    def mark_image_failed(self, image_name: str, reason: str) -> None:
        """Mark an image as failed with a reason"""
        self.failed_images[image_name] = reason
        if self.in_progress_image == image_name:
            self.in_progress_image = None
        
        # Update image metadata
        if image_name in self.image_metadata:
            self.image_metadata[image_name]["status"] = "failed"
            self.image_metadata[image_name]["failure_reason"] = reason
            self.image_metadata[image_name]["failed_at"] = datetime.now().isoformat()
        
        # Update dataset statistics
        self.update_dataset_statistics()
        
        # Save manifest after a failure
        self.save_manifest()
    
    def mark_patch_completed(self, image_name: str, x: int, y: int) -> None:
        """Mark a specific patch as completed"""
        if image_name not in self.completed_patches:
            self.completed_patches[image_name] = set()
        self.completed_patches[image_name].add(f"{x}_{y}")
    
    # --- Shard Tracking Methods ---
    
    def update_shard_info(self, prefix: str, split: str, index: int, size: int, count: int) -> None:
        """Update basic shard tracking information (compatible with original TilingCheckpoint)"""
        if prefix not in self.shard_indices:
            self.shard_indices[prefix] = {"trn": 0, "val": 0, "tst": 0}
            self.shard_sizes[prefix] = {"trn": 0, "val": 0, "tst": 0}
            self.patch_counts[prefix] = {"trn": 0, "val": 0, "tst": 0}
        
        self.shard_indices[prefix][split] = index
        self.shard_sizes[prefix][split] = size
        self.patch_counts[prefix][split] = count
        
        # Also update enhanced shard tracking
        self.update_shard_record(prefix, split, index, size, count)
    
    def update_shard_record(self, prefix: str, split: str, shard_index: int, 
                        size_bytes: int, patch_count: int, 
                        status: str = "OPEN", images: List[str] = None):
        """Create or update a detailed shard record"""
        # Ensure structure exists
        if split not in self.shards:
            self.shards[split] = []
        
        # Find shard if it exists
        for shard in self.shards[split]:
            if shard["id"] == shard_index:
                # Update existing shard - calculate incremental patch count
                prev_shard_total = 0
                for s in self.shards[split]:
                    if s["id"] < shard_index:
                        prev_shard_total += s.get("patch_count", 0)
                
                actual_shard_patches = patch_count - prev_shard_total
                
                # Update existing shard
                shard["size_bytes"] = size_bytes
                shard["patch_count"] = actual_shard_patches
                shard["status"] = status
                if images:
                    # Add new images to the shard's image list
                    if "images" not in shard:
                        shard["images"] = []
                    for img in images:
                        if img not in shard["images"]:
                            shard["images"].append(img)
                shard["last_updated"] = datetime.now().isoformat()
                return
        
        # Create new shard - calculate actual per-shard patch count
        prev_shard_total = 0
        for shard in self.shards[split]:
            if shard["id"] < shard_index:
                prev_shard_total += shard.get("patch_count", 0)
        
        actual_shard_patches = patch_count - prev_shard_total
        
        shard_path = f"{prefix}-{split}-{shard_index:06d}.tar"
        shard_entry = {
            "id": shard_index,
            "path": shard_path,
            "status": status,
            "size_bytes": size_bytes,
            "patch_count": actual_shard_patches,  # Use calculated per-shard count
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "images": images or []
        }
        self.shards[split].append(shard_entry)
        
        # Sort shards by ID for consistency
        self.shards[split] = sorted(self.shards[split], key=lambda s: s["id"])
    
    def get_shard_info(self, prefix: str, split: str) -> tuple:
        """Get current shard information (compatible with original TilingCheckpoint)"""
        if prefix not in self.shard_indices:
            return 0, 0, 0
        
        index = self.shard_indices[prefix].get(split, 0)
        size = self.shard_sizes[prefix].get(split, 0)
        count = self.patch_counts[prefix].get(split, 0)
        
        return index, size, count
    
    def close_shard(self, prefix: str, split: str, shard_index: int):
        """Mark a shard as closed (no more data can be added)"""
        for shard in self.shards[split]:
            if shard["id"] == shard_index:
                shard["status"] = "CLOSED"
                shard["closed_at"] = datetime.now().isoformat()
                break
    
    def find_open_shard(self, prefix: str, split: str, max_size_bytes: int = 2 * 1024 * 1024 * 1024):
        """Find an open shard for the given split with space available"""
        open_shards = []
        for shard in self.shards.get(split, []):
            if shard["status"] == "OPEN" and shard["size_bytes"] < max_size_bytes:
                open_shards.append(shard)
        
        if not open_shards:
            # No suitable shard found, create a new one
            next_index = self.get_next_shard_index(split)
            self.update_shard_record(prefix, split, next_index, 0, 0, "OPEN")
            return next_index
        
        # Return the shard with the most space available
        return min(open_shards, key=lambda s: s["size_bytes"])["id"]
    
    def get_next_shard_index(self, split: str) -> int:
        """Get the next available shard index for a split"""
        if not self.shards.get(split, []):
            return 0
        return max([s["id"] for s in self.shards.get(split, [])]) + 1
    
    # --- Image Metadata Methods ---
    
    def update_image_metadata(self, image_name: str, metadata: Dict[str, Any]):
        """Update or add metadata for an image"""
        if image_name not in self.image_metadata:
            self.image_metadata[image_name] = {
                "added_at": datetime.now().isoformat(),
                "status": "pending",
                "path": "",
                "label_path": "",
                "patches": {
                    "total": 0,
                    "distribution": {"trn": 0, "val": 0, "tst": 0},
                    "shard_locations": {"trn": [], "val": [], "tst": []}
                }
            }
        
        # Update metadata fields
        self.image_metadata[image_name].update(metadata)
        
        # Always update the last_updated timestamp
        self.image_metadata[image_name]["last_updated"] = datetime.now().isoformat()
    
    def update_image_patch_info(self, image_name: str, split: str, shard_index: int, patch_count: int = 1):
        """Update patch information for an image"""
        if image_name not in self.image_metadata:
            # Initialize with basic info if not exists
            self.update_image_metadata(image_name, {"status": "in_progress"})
        
        # Ensure patches structure exists
        if "patches" not in self.image_metadata[image_name]:
            self.image_metadata[image_name]["patches"] = {
                "total": 0,
                "distribution": {},
                "shard_locations": {}
            }
        
        patches = self.image_metadata[image_name]["patches"]
        
        # Update patch counts
        patches["total"] += patch_count
        
        if split not in patches["distribution"]:
            patches["distribution"][split] = 0
        patches["distribution"][split] += patch_count
        
        # Update shard locations
        if split not in patches["shard_locations"]:
            patches["shard_locations"][split] = []
        
        if shard_index not in patches["shard_locations"][split]:
            patches["shard_locations"][split].append(shard_index)
            # Keep locations sorted
            patches["shard_locations"][split].sort()
        
        # Update image in shard record
        self.add_image_to_shard(split, shard_index, image_name)
    
    def add_image_to_shard(self, split: str, shard_index: int, image_name: str):
        """Add an image to a shard's record"""
        for shard in self.shards[split]:
            if shard["id"] == shard_index:
                if "images" not in shard:
                    shard["images"] = []
                if image_name not in shard["images"]:
                    shard["images"].append(image_name)
                break
    
    # --- Dataset Statistics Methods ---
    
    def update_dataset_statistics(self):
        """Update overall dataset statistics"""
        # Update image counts
        self.dataset_statistics["image_counts"] = {
            "total": len(self.completed_images) + len(self.failed_images) + (1 if self.in_progress_image else 0),
            "completed": len(self.completed_images),
            "failed": len(self.failed_images),
            "in_progress": 1 if self.in_progress_image else 0
        }
        
        # Update patch counts
        patch_counts = {"trn": 0, "val": 0, "tst": 0}
        for prefix, counts in self.patch_counts.items():
            for split, count in counts.items():
                if split in patch_counts:
                    patch_counts[split] += count
        
        self.dataset_statistics["patch_counts"] = patch_counts
        
        # Update split ratios
        total_patches = sum(patch_counts.values())
        if total_patches > 0:
            self.dataset_statistics["actual_split_ratio"] = {
                split: count / total_patches 
                for split, count in patch_counts.items()
                if split in ["trn", "val", "tst"]  # Ensure only valid splits
            }
    
    def update_class_distribution(self, class_distribution: Dict[str, float]):
        """Update overall class distribution with proper weighting"""
        if not self.dataset_statistics["class_distribution"]:
            self.dataset_statistics["class_distribution"] = class_distribution
            self._class_update_count = 1
        else:
            # Use a counter instead of completed_images length
            if not hasattr(self, '_class_update_count'):
                self._class_update_count = 1
            
            # Weight old distribution by number of updates
            weight = self._class_update_count / (self._class_update_count + 1)
            
            new_distribution = {}
            for cls, value in class_distribution.items():
                old_value = self.dataset_statistics["class_distribution"].get(cls, 0)
                new_distribution[cls] = old_value * weight + value * (1 - weight)
            
            self.dataset_statistics["class_distribution"] = new_distribution
            self._class_update_count += 1
    
    def update_running_statistics(self, prefix: str, patch: np.ndarray):
        """
        Update running statistics with a new training patch.
        
        Args:
            prefix (str): Dataset prefix (determines sensor type)
            patch (np.ndarray): Image patch with shape (bands, height, width)
        """
        # Initialize on first patch
        if prefix not in self.running_statistics:
            self._initialize_band_statistics(prefix, patch)
        
        stats = self.running_statistics[prefix]
        
        # Validate band consistency
        if patch.shape[0] != stats["band_count"]:
            raise ValueError(
                f"Band count mismatch for prefix '{prefix}': "
                f"expected {stats['band_count']}, got {patch.shape[0]}"
            )
        
        # Convert to float64 for precision
        patch_float = patch.astype(np.float64)
        
        # Update statistics for each band
        for band_idx in range(stats["band_count"]):
            band_data = patch_float[band_idx].ravel()
            stats["band_sums"][band_idx] += np.sum(band_data)
            stats["band_sums_squared"][band_idx] += np.sum(band_data ** 2)
        
        # Update counts
        pixels_per_patch = patch.shape[1] * patch.shape[2]
        stats["pixel_count"] += pixels_per_patch
        stats["patch_count"] += 1
        stats["last_updated"] = datetime.now().isoformat()
        
        # Increment counter for save frequency
        self.stats_update_counter += 1
        
        # Save periodically
        if self.stats_update_counter >= self.stats_save_frequency:
            self.save_manifest()
            self.stats_update_counter = 0
            logger.debug(f"Saved manifest after {self.stats_save_frequency} patches")
    
    def get_dataset_statistics(self, prefix: str) -> Dict[str, Any]:
        """
        Calculate and return mean and standard deviation from running statistics.
        
        Args:
            prefix (str): Dataset prefix
            
        Returns:
            Dictionary with 'mean' and 'std' lists for each band, plus metadata
        """
        if prefix not in self.running_statistics:
            raise ValueError(f"No statistics found for prefix '{prefix}'")
        
        stats = self.running_statistics[prefix]
        
        if stats["pixel_count"] == 0:
            raise ValueError(f"No patches processed yet for prefix '{prefix}'")
        
        # Calculate mean and std for each band
        means = []
        stds = []
        
        for band_idx in range(stats["band_count"]):
            # Mean = sum / n
            mean = stats["band_sums"][band_idx] / stats["pixel_count"]
            
            # Variance = E[X^2] - E[X]^2
            mean_of_squares = stats["band_sums_squared"][band_idx] / stats["pixel_count"]
            variance = mean_of_squares - mean ** 2
            
            # Handle numerical precision issues
            variance = max(0, variance)
            std = np.sqrt(variance)
            
            means.append(mean)
            stds.append(std)
        
        return {
            "mean": means,
            "std": stds,
            "band_count": stats["band_count"],
            "pixel_count": stats["pixel_count"],
            "patch_count": stats["patch_count"],
            "dtype": stats["dtype"],
            "last_updated": stats["last_updated"]
        }
    def get_all_dataset_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all prefixes in the dataset."""
        all_stats = {}
        for prefix in self.running_statistics:
            try:
                all_stats[prefix] = self.get_dataset_statistics(prefix)
            except ValueError as e:
                logger.warning(f"Could not get statistics for {prefix}: {e}")
        return all_stats
    
    def is_split_ratio_drifting(self, threshold=0.03):
        """Check if the split ratio is drifting from target"""
        if "actual_split_ratio" not in self.dataset_statistics:
            return False
        
        actual = self.dataset_statistics["actual_split_ratio"]
        # Default target - could be made configurable
        target = {"trn": 0.8, "val": 0.2}
        
        # Check if drift exceeds threshold
        return abs(actual.get("trn", 0) - target["trn"]) > threshold
    
    def get_adjusted_val_ratio(self, default_ratio=0.2):
        """Get adjusted validation ratio to correct drift"""
        if not self.is_split_ratio_drifting():
            return default_ratio
        
        # Calculate adjustment
        actual = self.dataset_statistics["actual_split_ratio"]
        target = {"trn": 0.8, "val": 0.2}
        
        # If we have too many validation samples, reduce validation ratio
        if actual.get("val", 0) > target["val"]:
            return max(0.1, default_ratio - 0.05)  # Reduce by 5%
        else:
            return min(0.3, default_ratio + 0.05)  # Increase by 5%
    
    def get_total_sizes_by_split(self) -> Dict[str, int]:
        """Calculate total size across all shards for each split"""
        total_sizes = {"trn": 0, "val": 0, "tst": 0}
        
        for split in ["trn", "val", "tst"]:
            for shard in self.shards.get(split, []):
                total_sizes[split] += shard.get("size_bytes", 0)
        
        return total_sizes
    
    # --- Save and Load Methods ---
    
    def save_manifest(self):
        """Save manifest to disk"""
        # Update statistics before saving
        self.update_dataset_statistics()
        
        # Update last_updated timestamp
        self.dataset_info["last_updated"] = datetime.now().isoformat()
        
        # Prepare manifest data
        manifest_data = {
            "dataset_info": self.dataset_info,
            "shards": self.shards,
            "statistics": self.dataset_statistics,
            "running_statistics": self.running_statistics,
            "processed_images": self.image_metadata,
            "progress": {
                "completed_images": list(self.completed_images),
                "failed_images": self.failed_images,
                "in_progress_image": self.in_progress_image,
                "completed_patches": {
                    img: list(patches) for img, patches in self.completed_patches.items()
                },
                "shard_indices": self.shard_indices,
                "shard_sizes": self.shard_sizes,
                "patch_counts": self.patch_counts
            }
        }
        
        # Write to temporary file first (atomic write pattern)
        temp_path = self.manifest_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(manifest_data, f, indent=2, cls=NumpyEncoder)
        
        # Atomic rename
        temp_path.rename(self.manifest_path)
    
    def _load_manifest(self):
        """Load manifest from disk"""
        try:
            with open(self.manifest_path, 'r') as f:
                manifest_data = json.load(f, object_hook=numpy_decoder)
            
            # Load dataset info
            self.dataset_info = manifest_data.get("dataset_info", {})
            
            # Load shards data
            self.shards = manifest_data.get("shards", {"trn": [], "val": [], "tst": []})
            
            # Load statistics
            self.dataset_statistics = manifest_data.get("statistics", {})
            
            # Load running statistics
            self.running_statistics = manifest_data.get("running_statistics", {})
            self.stats_update_counter = 0 # Reset counter
            
            # Load image metadata
            self.image_metadata = manifest_data.get("processed_images", {})
            
            # Load progress data
            progress = manifest_data.get("progress", {})
            self.completed_images = set(progress.get("completed_images", []))
            self.failed_images = progress.get("failed_images", {})
            self.in_progress_image = progress.get("in_progress_image")
            
            # Load patch tracking data (convert lists back to sets)
            patch_data = progress.get("completed_patches", {})
            self.completed_patches = {
                img: set(patches) for img, patches in patch_data.items()
            }
            
            # Load shard tracking data
            self.shard_indices = progress.get("shard_indices", {})
            self.shard_sizes = progress.get("shard_sizes", {})
            self.patch_counts = progress.get("patch_counts", {})
            
            # Reset any in-progress image - it was interrupted
            self.in_progress_image = None
            
        except Exception as e:
            logger.error(f"Error loading manifest: {e}")
            # Initialize empty state if loading fails
            self._initialize_empty_state()
    
    def _initialize_empty_state(self):
        """Initialize empty state for all data structures"""
        self.completed_images = set()
        self.failed_images = {}
        self.in_progress_image = None
        self.completed_patches = {}
        self.shard_indices = {}
        self.shard_sizes = {}
        self.patch_counts = {}
        self.dataset_info = {
            "name": self.prefix,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "description": "Geospatial dataset for land cover classification"
        }
        self.shards = {"trn": [], "val": [], "tst": []}
        self.image_metadata = {}
        self.dataset_statistics = {
            "class_distribution": {},
            "patch_counts": {"trn": 0, "val": 0, "tst": 0},
            "image_counts": {"total": 0, "completed": 0, "failed": 0, "in_progress": 0},
            "actual_split_ratio": {"trn": 0, "val": 0, "tst": 0}
        }
        self.running_statistics = {}
        self.stats_update_counter = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manifest statistics"""
        return {
            "completed_images": len(self.completed_images),
            "failed_images": len(self.failed_images),
            "in_progress_image": self.in_progress_image,
            "patches_by_image": {
                img: len(patches) for img, patches in self.completed_patches.items()
            },
            "dataset_statistics": self.dataset_statistics
        }
    
    # --- Helper methods for continuous dataset growth ---
    
    def get_validation_ratio(self, default_ratio=0.2):
        """Get the appropriate validation ratio for new data"""
        if self.is_split_ratio_drifting():
            return self.get_adjusted_val_ratio(default_ratio)
        return default_ratio
    
    def get_full_manifest(self) -> Dict[str, Any]:
        """Get the complete manifest data structure"""
        self.update_dataset_statistics()
        
        return {
            "dataset_info": self.dataset_info,
            "shards": self.shards,
            "statistics": self.dataset_statistics,
            "processed_images": self.image_metadata,
            "progress": {
                "completed_images": list(self.completed_images),
                "failed_images": self.failed_images,
                "in_progress_image": self.in_progress_image,
                "completed_patches": {
                    img: list(patches) for img, patches in self.completed_patches.items()
                },
                "shard_indices": self.shard_indices,
                "shard_sizes": self.shard_sizes,
                "patch_counts": self.patch_counts
            }
        }
    
    def validate_manifest_consistency(self) -> Dict[str, Any]:
        """Validate consistency between different tracking systems (read-only)"""
        issues = []
        
        # Calculate totals from image metadata (source of truth)
        image_totals = {"trn": 0, "val": 0, "tst": 0}
        total_images = 0
        
        for image_name, image_meta in self.image_metadata.items():
            if image_meta.get("status") == "completed":
                total_images += 1
                distribution = image_meta.get("patches", {}).get("distribution", {})
                for split, count in distribution.items():
                    if split in image_totals:
                        image_totals[split] += count
        
        # Get totals from shards
        shard_totals = {"trn": 0, "val": 0, "tst": 0}
        for split in ["trn", "val", "tst"]:
            for shard in self.shards.get(split, []):
                shard_totals[split] += shard.get("patch_count", 0)
        
        # Get totals from statistics
        stats_totals = self.dataset_statistics.get("patch_counts", {})
        
        # Get running statistics count (training only)
        running_count = 0
        if self.running_statistics:
            for prefix_stats in self.running_statistics.values():
                running_count += prefix_stats.get("patch_count", 0)
        
        # Compare all tracking systems
        for split in ["trn", "val", "tst"]:
            image_count = image_totals[split]
            shard_count = shard_totals[split]
            stats_count = stats_totals.get(split, 0)
            
            if image_count != shard_count:
                issues.append(f"{split}: images={image_count} ≠ shards={shard_count}")
            
            if image_count != stats_count:
                issues.append(f"{split}: images={image_count} ≠ statistics={stats_count}")
        
        # Check running statistics (should match training patches)
        if running_count != image_totals["trn"]:
            issues.append(f"running_stats={running_count} ≠ training_patches={image_totals['trn']}")
        
        # Check completed image count
        completed_count = len(self.completed_images)
        if completed_count != total_images:
            issues.append(f"completed_images={completed_count} ≠ processed_images={total_images}")
        
        return {
            "is_consistent": len(issues) == 0,
            "issues": issues,
            "counts": {
                "from_images": image_totals,
                "from_shards": shard_totals, 
                "from_statistics": stats_totals,
                "from_running_stats": running_count,
                "completed_images": completed_count,
                "processed_images": total_images
            }
        }