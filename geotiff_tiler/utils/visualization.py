import matplotlib.pyplot as plt
import numpy as np
from utils.io import read_patches_from_zarr

def visualize_zarr_patches(zarr_path, 
                           indices=None, 
                           n_samples=3, 
                           figsize=(15, 10), 
                           cmap_image='viridis', 
                           cmap_label='viridis', 
                           save_path=None):
    """
    Visualize image and label patches from a zarr store.
    
    Args:
        zarr_path : str or Path
            Path to the zarr store
        indices : list or None
        Specific indices to visualize. If None, randomly selects n_samples
        n_samples : int
            Number of samples to visualize if indices is None
        figsize : tuple
            Figure size (width, height)
        cmap_image : str
            Colormap for image (only used for single-band images)
        cmap_label : str
            Colormap for label visualization
        save_path : str or None
            Path to save visualization. If None, doesn't save
    Returns:
        None
    """
    
    # Read patches from zarr
    image_patches, label_patches, patch_locations = read_patches_from_zarr(zarr_path)
    
    # Select indices to visualize
    total_patches = len(image_patches)
    if indices is None:
        if n_samples >= total_patches:
            indices = list(range(total_patches))
            n_samples = total_patches
        else:
            indices = np.random.choice(range(total_patches), size=n_samples, replace=False)
    else:
        indices = [i for i in indices if i < total_patches]
        n_samples = len(indices)
    
    if n_samples == 0:
        return None
    # Create figure
    fig, axes = plt.subplots(n_samples, 2, figsize=figsize)
    if n_samples == 1:
        axes = axes.reshape(1, 2)
    
    # Visualize each sample
    for i, idx in enumerate(indices):
        img = image_patches[idx]
        lbl = label_patches[idx]
        loc = patch_locations[idx]
        
        # Handle image display    
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] >= 3:
            img = img[:, :, :3]
            axes[i, 0].imshow(img)
        elif img.shape[2] < 3:
            img = img[:, :, :1]
            axes[i, 0].imshow(img, cmap=cmap_image)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        # Handle label display
        axes[i, 1].imshow(lbl[0], cmap=cmap_label)
        loc_str = f"({loc[0]}, {loc[1]})"
        # Add titles
        axes[i, 0].set_title(f"Image Patch {idx} at {loc_str}")
        axes[i, 1].set_title(f"Label Patch {idx}")
        
        # Remove axis ticks
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
