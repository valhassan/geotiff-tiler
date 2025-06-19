import logging
import matplotlib.pyplot as plt
import numpy as np
import webdataset as wds
from pathlib import Path
from typing import Tuple, Optional, Union
import random
from itertools import islice

logger = logging.getLogger(__name__)

def visualize_webdataset_patches(
    dataset_dir: Union[str, Path],
    prefix: str,
    split: str = "trn",
    image_name: Optional[str] = None,
    n_samples: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    cmap_label: str = 'tab10',
    save_path: Optional[str] = None,
    manifest = None
) -> None:
    """
    Visualize image and label patches from WebDataset tar files.
    
    Args:
        dataset_dir: Directory containing the dataset
        prefix: Dataset prefix
        split: Split to visualize
        image_name: Specific image to visualize (if None, samples randomly)
        n_samples: Number of patches to visualize
        figsize: Figure size
        cmap_label: Colormap for labels
        save_path: Path to save visualization
        manifest: Optional manifest for efficient shard lookup
    """
    dataset_path = Path(dataset_dir) / prefix / split
    
    # Efficient shard selection when looking for specific image
    if image_name and manifest:
        # Use manifest to find only shards containing this image
        image_meta = manifest.image_metadata.get(image_name, {})
        shard_locations = image_meta.get("patches", {}).get("shard_locations", {}).get(split, [])
        
        if not shard_locations:
            return
            
        # Build tar files list only for relevant shards
        tar_files = [
            dataset_path / f"{prefix}-{split}-{shard_id:06d}.tar"
            for shard_id in shard_locations
        ]
        tar_files = [f for f in tar_files if f.exists()]
        
        if not tar_files:
            return
            
    else:
        # Fall back to all tar files if no manifest or random sampling
        tar_files = sorted(dataset_path.glob(f"{prefix}-{split}-*.tar"))
        if not tar_files:
            return
    
    # Create WebDataset
    dataset = (wds.WebDataset([str(f) for f in tar_files])
               .decode()
               .to_tuple("image_patch.npy", "label_patch.npy", "metadata.json"))
    
    # Collect patches
    patches = []
    for img, lbl, metadata in dataset:
        patch_metadata = metadata.get("patch_metadata", {})
        
        # Filter by image name if specified
        if image_name and patch_metadata.get("image_name") != image_name:
            continue
        
        patches.append({
            "image": img,
            "label": lbl,
            "metadata": metadata,
            "coords": patch_metadata.get("pixel_coordinates", [0, 0]),
            "sensor": patch_metadata.get("sensor_type", "unknown")
        })
        
        # Early stopping
        if len(patches) >= n_samples * 2:  # Collect extra for random sampling
            break
    
    if not patches:
        return
    
    # Random sample if we have more than needed
    if len(patches) > n_samples:
        patches = random.sample(patches, n_samples)
    
    # Create figure
    n_patches = len(patches)
    fig, axes = plt.subplots(n_patches, 2, figsize=figsize)
    if n_patches == 1:
        axes = axes.reshape(1, 2)
    
    # Visualize each patch
    for i, patch_data in enumerate(patches):
        img = patch_data["image"]
        lbl = patch_data["label"]
        coords = patch_data["coords"]
        sensor = patch_data["sensor"]
        
        # Handle image display (CHW to HWC)
        img_display = np.transpose(img, (1, 2, 0))
        
        # RGB display
        if img_display.shape[2] >= 3:
            img_rgb = img_display[:, :, :3]
            # Normalize if needed
            if img_rgb.dtype in [np.uint8, np.uint16]:
                img_rgb = img_rgb.astype(np.float32) / np.iinfo(img_rgb.dtype).max
            axes[i, 0].imshow(img_rgb)
        else:
            # Single band
            axes[i, 0].imshow(img_display[:, :, 0], cmap='gray')
        
        # Label display
        lbl_display = lbl[0] if lbl.ndim == 3 else lbl
        axes[i, 1].imshow(lbl_display, cmap=cmap_label, interpolation='nearest')
        
        # Add titles with more info
        loc_str = f"({coords[0]}, {coords[1]})"
        axes[i, 0].set_title(f"{sensor}\nPatch {i} at {loc_str}", fontsize=10)
        axes[i, 1].set_title(f"Label Patch {i}\nClasses: {len(np.unique(lbl_display))}", fontsize=10)
        
        # Remove ticks
        for ax in [axes[i, 0], axes[i, 1]]:
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add image name to main title if specified
    if image_name:
        plt.suptitle(f"Patches from: {image_name}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_dataset_summary_visualization(
    output_dir: str,
    prefix: str,
    samples_per_split: int = 3,
    images_per_split: int = 3,
    max_shards_to_read: int = 3) -> None:
    """
    Create summary visualizations showing samples from multiple images across splits.
    
    Args:
        output_dir: Directory containing the dataset
        prefix: Dataset prefix
        samples_per_split: Total number of sample patches to show per split
        images_per_split: Number of different images to sample from per split
        max_shards_to_read: Maximum number of shards to read per split
    """
    
    # First, create individual visualizations for each split
    for split in ["trn", "val", "tst"]:
        dataset_path = Path(output_dir) / prefix / split
        tar_files = sorted(dataset_path.glob(f"{prefix}-{split}-*.tar"))
        
        if not tar_files:
            logger.debug(f"No tar files found for {split} split")
            continue
        
        if len(tar_files) > max_shards_to_read:
            indices = np.linspace(0, len(tar_files) - 1, max_shards_to_read, dtype=int)
            selected_tar_files = [tar_files[i] for i in indices]
        else:
            selected_tar_files = tar_files
        
        logger.debug(f"Reading {len(selected_tar_files)} shards for {split} split")
        
        image_names = set()
        patches_by_image = {}
        
        try:
            max_samples_to_read = images_per_split * 10000
            
            dataset = (wds.WebDataset([str(f) for f in selected_tar_files])
                       .decode()
                       .to_tuple("image_patch.npy", "label_patch.npy", "metadata.json"))
            
            for img, lbl, metadata in islice(dataset, max_samples_to_read):
                patch_metadata = metadata.get("patch_metadata", {})
                image_name = patch_metadata.get("image_name", "unknown")
                
                if image_name not in patches_by_image:
                    patches_by_image[image_name] = []
                    image_names.add(image_name)
                
                patches_by_image[image_name].append({
                    "image": img,
                    "label": lbl,
                    "metadata": metadata
                })
                
                if len(image_names) >= images_per_split * 2: 
                    break
            
        except Exception as e:
            logger.error(f"Error reading {split} split: {e}")
            continue
        selected_images = list(image_names)[:images_per_split]
        if not selected_images:
            logger.debug(f"No images found in {split} split")
            continue
            
        if len(selected_images) < images_per_split:
            logger.debug(f"Warning: Only found {len(selected_images)} unique images in {split} split")
        
        patches_per_image = max(1, samples_per_split // len(selected_images))
        extra_patches = samples_per_split % len(selected_images)
        
        final_patches = []
        for idx, image_name in enumerate(selected_images):
            n_patches = patches_per_image + (1 if idx < extra_patches else 0)
            image_patches = patches_by_image.get(image_name, [])
            
            if not image_patches:
                continue
                
            sampled = random.sample(image_patches, min(n_patches, len(image_patches)))
            final_patches.extend(sampled)
        
        if not final_patches:
            logger.debug(f"No patches collected for {split} split")
            continue
        
        n_rows = len(final_patches)
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, 2)
        
        # Visualize collected patches
        for row_idx, patch_data in enumerate(final_patches):
            img = patch_data["image"]
            lbl = patch_data["label"]
            metadata = patch_data["metadata"]
            
            img_display = np.transpose(img, (1, 2, 0))
            if img_display.shape[2] >= 3:
                img_rgb = img_display[:, :, :3]
                if img_rgb.dtype in [np.uint8, np.uint16]:
                    img_rgb = img_rgb.astype(np.float32) / np.iinfo(img_rgb.dtype).max
                axes[row_idx, 0].imshow(img_rgb)
            else:
                axes[row_idx, 0].imshow(img_display[:, :, 0], cmap='gray')
            
            lbl_display = lbl[0] if lbl.ndim == 3 else lbl
            axes[row_idx, 1].imshow(lbl_display, cmap='tab10', interpolation='nearest')
            
            patch_metadata = metadata.get("patch_metadata", {})
            coords = patch_metadata.get("pixel_coordinates", [0, 0])
            image_name = patch_metadata.get("image_name", "unknown")
            sensor = patch_metadata.get("sensor_type", "unknown")
            
            axes[row_idx, 0].set_title(f"{image_name}\n{sensor} @ ({coords[0]}, {coords[1]})", 
                                      fontsize=9)
            axes[row_idx, 1].set_title(f"Label\nClasses: {len(np.unique(lbl_display))}", 
                                      fontsize=9)
            for ax in [axes[row_idx, 0], axes[row_idx, 1]]:
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle(f"{prefix} - {split.upper()} Split ({len(selected_images)} images)", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = Path(output_dir) / prefix / f"{prefix}_{split}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved visualization for {split} split: {save_path}")
    
    return None