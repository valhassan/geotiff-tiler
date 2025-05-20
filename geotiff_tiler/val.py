import math
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


def calculate_class_distribution(label, class_ids):
    """
    Calculate the distribution of classes in a label mask.
    
    Args:
        label: Label array (either rasterio dataset or numpy array)
        class_ids: Dictionary mapping class names to pixel values
    
    Returns:
        Dictionary with class distribution (percentage of pixels for each class)
    """
    # Read data if it's a rasterio dataset
    if hasattr(label, 'read'):
        label_data = label.read()
    else:
        label_data = label
    
    # Count pixels for each class
    class_counts = {}
    for class_name, class_id in class_ids.items():
        class_counts[class_name] = np.sum(label_data == class_id)
    
    # Calculate distribution
    total_pixels = np.sum(list(class_counts.values()))
    if total_pixels > 0:
        distribution = {cls: count/total_pixels for cls, count in class_counts.items()}
    else:
        distribution = {cls: 0 for cls in class_ids}
    
    return distribution

def create_spatial_grid(image, stride_size, grid_size):
    """
    Create a spatial grid over an image and calculate statistics for each cell.
    
    Args:
        image: Image dataset (rasterio)
        label: Label dataset (rasterio)
        grid_size: Number of grid cells in each dimension
    
    Returns:
        Dictionary with grid information
    """
    
    minx, miny = 0, 0
    maxx, maxy = image.width, image.height
    
    stridex = maxx / grid_size
    stridey = maxy / grid_size
    print(f"stride_size: {stride_size}")
    number_of_patches_x = math.ceil(maxx / stride_size)
    number_of_patches_y = math.ceil(maxy / stride_size)
    total_patches = number_of_patches_x * number_of_patches_y
    # Initialize grid cells
    grid_cells = {}
    
    # Return grid info
    return {
        'minx': minx,
        'miny': miny,
        'maxx': maxx,
        'maxy': maxy,
        'stridex': stridex,
        'stridey': stridey,
        'grid_size': grid_size,
        'cells': grid_cells,
        'total_patches': total_patches
    }

def is_valid_pair(image, label):
    """
    Check if image and label are valid for processing.
    
    Args:
        image: Image dataset (rasterio)
        label: Label dataset (rasterio)
    
    Returns:
        Boolean indicating if the pair is valid
    """
    # Check if both are loaded
    if image is None or label is None:
        return False
    
    # Check dimensions match
    if hasattr(image, 'width') and hasattr(label, 'width'):
        if image.width != label.width or image.height != label.height:
            return False
    
    # Check if both are georeferenced
    if hasattr(image, 'crs') and hasattr(label, 'crs'):
        if image.crs is None or label.crs is None:
            return False
    
    return True

def select_validation_cells(grid, target_distribution, val_ratio, class_balance_weight, spatial_weight):
    """
    Select grid cells for validation based on class balance and spatial coverage.
    
    Args:
        grid: Grid information from create_spatial_grid
        target_distribution: Target class distribution
        val_ratio: Percentage of data to use for validation
        class_balance_weight: Weight for class balance score
        spatial_weight: Weight for spatial coverage score
    
    Returns:
        Set of cell IDs selected for validation
    """
    grid_size = grid['grid_size']
    total_patches = grid['total_patches']
    
    # Calculate target validation size
    target_val_size = max(5, int(total_patches * val_ratio))
    
    # Calculate cell scores
    cell_scores = {}
    
    # For each cell in the grid
    for grid_x in range(grid_size):
        for grid_y in range(grid_size):
            cell_id = f"{grid_x}_{grid_y}"
            
            # For spatial coverage, prefer variety (cells away from center)
            # This is a simplification - you could use actual cell statistics
            norm_x = grid_x / (grid_size - 1)  # Normalize to 0-1
            norm_y = grid_y / (grid_size - 1)  # Normalize to 0-1
            
            # Distance from center (0.5, 0.5)
            spatial_score = max(0.1, (norm_x - 0.5)**2 + (norm_y - 0.5)**2)
            
            # Class balance score - in a production version, would use actual cell statistics
            # For simplicity, assume uniform class distribution in this example
            class_score = 0.5  # Default medium score
            
            # Combined score (lower is better for selection)
            cell_scores[cell_id] = (class_balance_weight * class_score + spatial_weight * spatial_score)
    
    # Sort cells by score (lower is better)
    sorted_cells = sorted(cell_scores.items(), key=lambda x: x[1])
    
    # Select cells until we reach target size
    # In a real implementation, you'd track the actual number of patches in each cell
    validation_cells = set()
    estimated_val_size = 0
    
    # Estimate patches per cell (evenly distributed for simplicity)
    patches_per_cell = total_patches / (grid_size * grid_size)
    
    for cell_id, _ in sorted_cells:
        validation_cells.add(cell_id)
        estimated_val_size += patches_per_cell
        
        if estimated_val_size >= target_val_size:
            break
    
    return validation_cells

def create_shard_manifests(train_dir, val_dir):
    """
    Create manifest files listing all shards for train and validation.
    
    Args:
        train_dir: Directory with training shards
        val_dir: Directory with validation shards
    """
    # Find all training shards
    train_shards = []
    for sensor_dir in train_dir.glob("*"):
        if sensor_dir.is_dir():
            train_shards.extend(list(sensor_dir.glob("*.tar")))
    
    # Find all validation shards
    val_shards = []
    for sensor_dir in val_dir.glob("*"):
        if sensor_dir.is_dir():
            val_shards.extend(list(sensor_dir.glob("*.tar")))
    
    # Write train manifest
    with open(train_dir / "train-shards.txt", "w") as f:
        for shard in sorted(train_shards):
            f.write(f"{shard}\n")
    
    # Write validation manifest
    with open(val_dir / "val-shards.txt", "w") as f:
        for shard in sorted(val_shards):
            f.write(f"{shard}\n")

def create_validation_report(image_analyses, validation_cells, target_distribution, output_dir):
    """
    Create a report with visualizations of the validation split.
    
    Args:
        image_analyses: List of image analysis results
        validation_cells: Dictionary mapping image names to sets of validation cell IDs
        target_distribution: Target class distribution
        output_dir: Base output directory
    """
    report_dir = Path(output_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary report
    with open(report_dir / "validation_summary.txt", "w") as f:
        f.write(f"Validation Split Summary\n")
        f.write(f"=====================\n\n")
        f.write(f"Created: {datetime.now().isoformat()}\n\n")
        
        f.write(f"Target Class Distribution:\n")
        for cls, val in target_distribution.items():
            f.write(f"  - {cls}: {val:.4f}\n")
        
        f.write("\nImages Processed:\n")
        for analysis in image_analyses:
            f.write(f"\n{analysis['image_name']}:\n")
            f.write(f"  - Sensor: {analysis['sensor_type']}\n")
            
            # Add more details as needed
    
    # Create spatial visualizations for each image
    for analysis in image_analyses:
        image_name = analysis['image_name']
        grid = analysis['grid']
        
        # Create grid visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid
        for i in range(grid['grid_size'] + 1):
            ax.axvline(grid['minx'] + i * grid['stridex'], color='gray', linestyle='--', alpha=0.3)
            ax.axhline(grid['miny'] + i * grid['stridey'], color='gray', linestyle='--', alpha=0.3)
        
        # Highlight validation cells
        for cell_id in validation_cells.get(image_name, set()):
            grid_x, grid_y = map(int, cell_id.split('_'))
            
            # Calculate cell bounds
            cell_minx = grid['minx'] + grid_x * grid['stridex']
            cell_miny = grid['miny'] + grid_y * grid['stridey']
            
            # Draw cell
            rect = plt.Rectangle(
                (cell_minx, cell_miny), 
                grid['stridex'], grid['stridey'],
                linewidth=1, edgecolor='r', facecolor='r', alpha=0.2
            )
            ax.add_patch(rect)
        
        ax.set_title(f"Validation Grid - {image_name}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        plt.tight_layout()
        plt.savefig(report_dir / f"{image_name}_grid.png", dpi=150)
        plt.close()