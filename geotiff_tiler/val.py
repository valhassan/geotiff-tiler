import math
import logging
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

logger = logging.getLogger(__name__)

def calculate_class_distribution(label_path, class_ids):
    """
    Ultra-fast approach using bincount (works when class_ids are small integers).
    """
    with rasterio.open(label_path) as src_label:
        label_arr = src_label.read(1).flatten()
        
        # Use bincount for counting
        max_id = max(class_ids.values()) if class_ids else 0
        counts = np.bincount(label_arr, minlength=max_id + 1)
        
        # Extract counts for our classes
        class_counts = {}
        total_pixels = 0
        
        for class_name, class_id in class_ids.items():
            count = counts[class_id] if class_id < len(counts) else 0
            class_counts[class_name] = count
            total_pixels += count
        
        # Calculate distribution
        if total_pixels > 0:
            distribution = {cls: count/total_pixels for cls, count in class_counts.items()}
        else:
            distribution = {cls: 0 for cls in class_ids}
            
        return distribution

def create_spatial_grid(image_path, label_path, stride_size, grid_size, class_ids):
    """
    Ultra-fast vectorized version using numpy operations.
    """
    # Get dimensions
    with rasterio.open(image_path) as src_image:
        maxx, maxy = src_image.width, src_image.height
    
    # Read label data once
    with rasterio.open(label_path) as src_label:
        label_arr = src_label.read(1)
    
    minx, miny = 0, 0
    stridex = maxx / grid_size
    stridey = maxy / grid_size
    
    grid_cells = {}
    
    # Pre-allocate arrays for vectorized operations
    class_id_array = np.array(list(class_ids.values()))
    class_names = list(class_ids.keys())
    
    for grid_x in range(grid_size):
        for grid_y in range(grid_size):
            cell_minx = int(grid_x * stridex)
            cell_miny = int(grid_y * stridey)
            cell_maxx = int(min((grid_x + 1) * stridex, maxx))
            cell_maxy = int(min((grid_y + 1) * stridey, maxy))
            
            # Extract cell
            cell_data = label_arr[cell_miny:cell_maxy, cell_minx:cell_maxx]
            
            if cell_data.size == 0:
                distribution = {cls: 0 for cls in class_names}
            else:
                # Vectorized counting
                class_masks = cell_data[..., np.newaxis] == class_id_array
                class_counts = np.sum(class_masks, axis=(0, 1))
                
                # Convert to distribution
                total_pixels = cell_data.size
                distribution = {
                    name: count / total_pixels 
                    for name, count in zip(class_names, class_counts)
                }
            
            cell_id = f"{grid_x}_{grid_y}"
            grid_cells[cell_id] = {
                'distribution': distribution,
                'bounds': (cell_minx, cell_miny, cell_maxx, cell_maxy),
                'pixel_count': cell_data.size
            }
    
    # Calculate patch info
    number_of_patches_x = math.ceil(maxx / stride_size)
    number_of_patches_y = math.ceil(maxy / stride_size)
    total_patches = number_of_patches_x * number_of_patches_y
    
    return {
        'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy,
        'stridex': stridex, 'stridey': stridey, 'grid_size': grid_size,
        'cells': grid_cells, 'total_patches': total_patches
    }

def calculate_spatial_penalty(cell_data, validation_cells):
    """Calculate spatial penalty for a cell (lower penalty = better spatial diversity)."""
    if not validation_cells:
        # For first cell, prefer corners/edges for better initial coverage
        grid_x, grid_y = cell_data['grid_x'], cell_data['grid_y']
        # Assume grid_size is available or pass it as parameter
        # For now, estimate from coordinates (this could be improved)
        max_coord = max(grid_x, grid_y)
        if max_coord == 0:
            return 0  # Corner cell
        
        # Prefer cells further from center
        center_dist = abs(grid_x - max_coord/2) + abs(grid_y - max_coord/2)
        return -center_dist  # Negative because we want distance from center
    
    # Calculate minimum distance to any existing validation cell
    min_distance = float('inf')
    grid_x, grid_y = cell_data['grid_x'], cell_data['grid_y']
    
    for selected_id in validation_cells:
        # Parse coordinates from cell_id
        sel_x, sel_y = map(int, selected_id.split('_'))
        # Use Manhattan distance for efficiency
        distance = abs(grid_x - sel_x) + abs(grid_y - sel_y)
        min_distance = min(min_distance, distance)
    
    # Return negative distance so that larger distances give lower penalties
    return -min_distance

def select_validation_cells(grid, target_distribution, val_ratio, class_balance_weight, spatial_weight):
    """Select grid cells for validation with guaranteed class coverage and spatial diversity."""
    grid_size = grid['grid_size']
    total_patches = grid['total_patches']
    
    # Calculate target validation size
    target_val_size = max(5, int(total_patches * val_ratio))
    
    logger.debug(f"Selecting validation cells: target_val_size={target_val_size}, "
                f"total_patches={total_patches}, val_ratio={val_ratio}")
    
    # Pre-calculate cell information for efficiency
    valid_cells = {}
    cells_by_class = {cls: [] for cls in target_distribution.keys()}
    
    for grid_x in range(grid_size):
        for grid_y in range(grid_size):
            cell_id = f"{grid_x}_{grid_y}"
            cell_info = grid['cells'].get(cell_id)
            
            if not cell_info:
                continue
            
            # Calculate class balance score (L1 distance from target)
            class_score = sum(abs(cell_info['distribution'].get(cls, 0) - target_distribution[cls]) 
                            for cls in target_distribution)
            
            # Store cell info
            valid_cells[cell_id] = {
                'grid_x': grid_x,
                'grid_y': grid_y,
                'class_score': class_score,
                'distribution': cell_info['distribution']
            }
            
            # Index cells by classes they contain (with meaningful presence)
            for cls, proportion in cell_info['distribution'].items():
                if proportion > 0.01:  # Threshold for meaningful class presence
                    cells_by_class[cls].append(cell_id)
    
    validation_cells = set()
    patches_per_cell = total_patches / (grid_size * grid_size)
    estimated_val_size = 0
    
    # Phase 1: Ensure all classes are represented
    logger.debug("Phase 1: Ensuring class coverage")
    required_classes = set(target_distribution.keys())
    covered_classes = set()
    
    while covered_classes != required_classes and estimated_val_size < target_val_size:
        missing_classes = required_classes - covered_classes
        best_cell = None
        best_score = float('inf')
        
        # Find the best cell that covers missing classes
        for cell_id in valid_cells:
            if cell_id in validation_cells:
                continue
            
            cell_data = valid_cells[cell_id]
            
            # Check which missing classes this cell covers
            cell_classes = {cls for cls, val in cell_data['distribution'].items() if val > 0.01}
            new_classes = cell_classes & missing_classes
            
            if not new_classes:
                continue
            
            # Score: prioritize cells that cover many missing classes
            class_coverage_bonus = -len(new_classes) * 10  # Heavy bonus for covering more classes
            class_balance_penalty = cell_data['class_score']
            spatial_penalty = calculate_spatial_penalty(cell_data, validation_cells)
            
            score = class_coverage_bonus + class_balance_weight * class_balance_penalty + spatial_weight * spatial_penalty
            
            if score < best_score:
                best_score = score
                best_cell = cell_id
        
        if best_cell:
            validation_cells.add(best_cell)
            estimated_val_size += patches_per_cell
            
            # Update covered classes
            cell_classes = {cls for cls, val in valid_cells[best_cell]['distribution'].items() if val > 0.01}
            covered_classes.update(cell_classes)
            
            logger.debug(f"Phase 1: Added {best_cell}, covered classes: {covered_classes}")
        else:
            logger.debug(f"Could not find cells for missing classes: {missing_classes}")
            break
    
    # Phase 2: Fill remaining slots with spatial diversity + class balance
    logger.debug("Phase 2: Optimizing spatial diversity and class balance")
    
    while estimated_val_size < target_val_size and len(validation_cells) < len(valid_cells):
        best_cell = None
        best_score = float('inf')
        
        for cell_id in valid_cells:
            if cell_id in validation_cells:
                continue
            
            cell_data = valid_cells[cell_id]
            
            # Spatial penalty (want maximum distance to existing cells)
            spatial_penalty = calculate_spatial_penalty(cell_data, validation_cells)
            
            # Class balance penalty
            class_balance_penalty = cell_data['class_score']
            
            # Combined score (lower is better)
            score = class_balance_weight * class_balance_penalty + spatial_weight * spatial_penalty
            
            if score < best_score:
                best_score = score
                best_cell = cell_id
        
        if best_cell:
            validation_cells.add(best_cell)
            estimated_val_size += patches_per_cell
            
            if len(validation_cells) <= 5:
                logger.debug(f"Phase 2: Selected cell {best_cell} with score {best_score:.3f}")
        else:
            break
    
    # Final verification
    final_covered_classes = set()
    for cell_id in validation_cells:
        cell_classes = {cls for cls, val in valid_cells[cell_id]['distribution'].items() if val > 0.01}
        final_covered_classes.update(cell_classes)
    
    missing_classes = required_classes - final_covered_classes
    if missing_classes:
        logger.debug(f"Final validation set missing classes: {missing_classes}")
    else:
        logger.debug("All classes represented in validation set")
    
    logger.debug(f"Selected {len(validation_cells)} validation cells covering {len(final_covered_classes)} classes")
    logger.debug(f"Selected cells: {sorted(validation_cells)}")
    
    return validation_cells

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