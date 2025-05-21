import os
import json
import logging
import signal
from pathlib import Path
from typing import Dict, Any, Set, Optional

class TilingCheckpoint:
    """
    Simple checkpoint manager for tiling operations.
    
    Tracks processed images, patches, and shard information to enable
    resuming tiling operations after interruption.
    """
    
    def __init__(self, output_dir: str, prefix: str):
        """
        Initialize the checkpoint manager.
        
        Args:
            output_dir (str): Directory for output files and checkpoint
            prefix (str): Dataset prefix
        """
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.checkpoint_path = self.output_dir / f"{self.prefix}_checkpoint.json"
        
        # Track completed images and patches
        self.completed_images: Set[str] = set()
        self.failed_images: Dict[str, str] = {}  # image_name -> reason
        self.in_progress_image: Optional[str] = None
        self.completed_patches: Dict[str, Set[str]] = {}  # image_name -> set of "x_y" strings
        
        # Track shard information
        self.shard_indices: Dict[str, Dict[str, int]] = {}  # prefix -> {split -> index}
        self.shard_sizes: Dict[str, Dict[str, int]] = {}    # prefix -> {split -> size}
        self.patch_counts: Dict[str, Dict[str, int]] = {}   # prefix -> {split -> count}
        
        # Load checkpoint if resuming
        if self.checkpoint_path.exists():
            self._load_checkpoint()
            logging.info(f"Resumed from checkpoint: {self.checkpoint_path}")
        
        # Register signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Set up handlers to save checkpoint on interrupt"""
        def handle_interrupt(signum, frame):
            logging.info("Interrupt received, saving checkpoint...")
            self.save_checkpoint()
            # Re-raise to allow normal exit processing
            raise KeyboardInterrupt("Tiling process was interrupted")
        
        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_interrupt)
    
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
    
    def mark_image_completed(self, image_name: str) -> None:
        """Mark an image as completely processed"""
        self.completed_images.add(image_name)
        if image_name in self.failed_images:
            del self.failed_images[image_name]
        if self.in_progress_image == image_name:
            self.in_progress_image = None
        # Save checkpoint after completing an image
        self.save_checkpoint()
    
    def mark_image_failed(self, image_name: str, reason: str) -> None:
        """Mark an image as failed with a reason"""
        self.failed_images[image_name] = reason
        if self.in_progress_image == image_name:
            self.in_progress_image = None
        # Save checkpoint after a failure
        self.save_checkpoint()
    
    def mark_patch_completed(self, image_name: str, x: int, y: int) -> None:
        """Mark a specific patch as completed"""
        if image_name not in self.completed_patches:
            self.completed_patches[image_name] = set()
        self.completed_patches[image_name].add(f"{x}_{y}")
    
    def update_shard_info(self, prefix: str, split: str, index: int, size: int, count: int) -> None:
        """Update shard tracking information"""
        if prefix not in self.shard_indices:
            self.shard_indices[prefix] = {"trn": 0, "val": 0, "tst": 0}
            self.shard_sizes[prefix] = {"trn": 0, "val": 0, "tst": 0}
            self.patch_counts[prefix] = {"trn": 0, "val": 0, "tst": 0}
        
        self.shard_indices[prefix][split] = index
        self.shard_sizes[prefix][split] = size
        self.patch_counts[prefix][split] = count
    
    def get_shard_info(self, prefix: str, split: str) -> tuple:
        """Get current shard information"""
        if prefix not in self.shard_indices:
            return 0, 0, 0
        
        index = self.shard_indices[prefix].get(split, 0)
        size = self.shard_sizes[prefix].get(split, 0)
        count = self.patch_counts[prefix].get(split, 0)
        
        return index, size, count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        return {
            "completed_images": len(self.completed_images),
            "failed_images": len(self.failed_images),
            "in_progress_image": self.in_progress_image,
            "patches_by_image": {
                img: len(patches) for img, patches in self.completed_patches.items()
            }
        }
    
    def save_checkpoint(self) -> None:
        """Save checkpoint to disk"""
        # Ensure directory exists
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint_data = {
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
        
        # Write to temporary file first (atomic write pattern)
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Atomic rename
        temp_path.rename(self.checkpoint_path)
    
    def _load_checkpoint(self) -> None:
        """Load checkpoint from disk"""
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Load image tracking data
            self.completed_images = set(checkpoint_data.get("completed_images", []))
            self.failed_images = checkpoint_data.get("failed_images", {})
            self.in_progress_image = checkpoint_data.get("in_progress_image")
            
            # Load patch tracking data (convert lists back to sets)
            patch_data = checkpoint_data.get("completed_patches", {})
            self.completed_patches = {
                img: set(patches) for img, patches in patch_data.items()
            }
            
            # Load shard tracking data
            self.shard_indices = checkpoint_data.get("shard_indices", {})
            self.shard_sizes = checkpoint_data.get("shard_sizes", {})
            self.patch_counts = checkpoint_data.get("patch_counts", {})
            
            # Reset any in-progress image - it was interrupted
            self.in_progress_image = None
            
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            # Initialize empty state if loading fails
            self.completed_images = set()
            self.failed_images = {}
            self.in_progress_image = None
            self.completed_patches = {}
            self.shard_indices = {}
            self.shard_sizes = {}
            self.patch_counts = {}