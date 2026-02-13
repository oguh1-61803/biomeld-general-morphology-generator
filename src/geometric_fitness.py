"""
Geometric Fitness Module for CPPN Morphology Pre-Evolution

Calculates shape-based fitness scores without physics simulation.
Uses universal metrics with archetype-specific weight combinations.

Date: December 2025
Version: 1.0
"""

import numpy as np
from scipy.ndimage import label
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')  # Suppress numpy warnings for cleaner output


# ============================================================================
# WEIGHT PRESETS FOR TARGET SHAPES
# ============================================================================

SHAPE_WEIGHT_PRESETS = {
    'cylinder': {
        'aspect_ratio': 0.25,
        'radial_symmetry': 0.25,
        'circular_symmetry': 0.25,
        'elongation': 0.20,
        'connectivity': 0.15,
        'hollowness': 0.10,
        'fill_ratio': 0.05,
        # Unused metrics get 0.0 weight automatically
    },
    
    'sphere': {
        'compactness': 0.30,
        'spherical_symmetry': 0.25,
        'aspect_ratio': 0.20,  # Should be close to 1.0
        'connectivity': 0.15,
        'fill_ratio': 0.10,
    },
    
    'cube': {
        'aspect_ratio': 0.25,  # Should be close to 1.0
        'bilateral_symmetry': 0.25,
        'compactness': 0.20,
        'connectivity': 0.15,
        'fill_ratio': 0.15,
    },
    
    'ellipsoid': {
        'elongation': 0.25,
        'spherical_symmetry': 0.20,
        'compactness': 0.20,
        'aspect_ratio': 0.15,
        'connectivity': 0.10,
        'fill_ratio': 0.10,
    }
}

# Target values for metrics (used for scoring)
TARGET_VALUES = {
    'cylinder': {
        'aspect_ratio': (15.0, 25.0),  # Range for cylinders
        'radial_symmetry': 0.80,        # High radial symmetry
        'circular_symmetry': 0.80,        # High radial symmetry
        'elongation': 8.0,              # Highly elongated
        'hollowness': 0.60,             # Moderately hollow
    },
    
    'sphere': {
        'aspect_ratio': (0.9, 1.1),    # Nearly 1:1:1
        'compactness': 0.90,            # Maximum compactness
        'spherical_symmetry': 0.85,     # High spherical symmetry
    },
    
    'cube': {
        'aspect_ratio': (0.9, 1.1),    # 1:1:1 ratio
        'bilateral_symmetry': 0.80,     # Three mirror planes
        'compactness': 0.70,            # Moderately compact
    },
    
    'ellipsoid': {
        'aspect_ratio': (1.5, 3.0),    # Stretched sphere
        'elongation': 3.0,              # Moderately elongated
        'spherical_symmetry': 0.65,     # Approximate spherical
        'compactness': 0.75,            # Fairly compact
    }
}


# ============================================================================
# MAIN FITNESS FUNCTION
# ============================================================================

def calculate_fitness(morphology_array: np.ndarray, 
                     target_shape: str,
                     return_details: bool = False) -> float:
    """
    Calculate geometric fitness score for a morphology.
    
    Args:
        morphology_array: 3D numpy array (Z, Y, X) with voxel values
                         0 = empty, 1 = passive, 3 = active
        target_shape: Target shape name ('cylinder', 'sphere', 'cube', 'ellipsoid')
        return_details: If True, return (fitness, metrics_dict, scores_dict)
    
    Returns:
        float: Fitness score [0, 1], or tuple if return_details=True
    
    Example:
        >>> morph = np.random.randint(0, 4, (10, 8, 15))
        >>> fitness = calculate_fitness(morph, 'cylinder')
        >>> print(f"Fitness: {fitness:.3f}")
    """
    #print("AAAAAAAAAA")
    # Validate inputs
    if target_shape not in SHAPE_WEIGHT_PRESETS:
        raise ValueError(
            f"Unknown target shape: {target_shape}. "
            f"Must be one of: {list(SHAPE_WEIGHT_PRESETS.keys())}"
        )
    #print("AAAAAAAAAA2")    
    # Get weight preset for this shape
    weights = SHAPE_WEIGHT_PRESETS[target_shape]
    targets = TARGET_VALUES.get(target_shape, {})
    
    # Calculate all metrics
    metrics = calculate_all_metrics(morphology_array)
    #print("AAAAAAAAAA3")   
    # Check hard constraints (fatal failures)
    if not check_hard_constraints(metrics):
        if return_details:
            return 0.0, metrics, {}
        return 0.0
    #print("AAAAAAAAAA4")   
    # Score each metric (normalize to [0, 1])
    scores = score_metrics(metrics, targets, target_shape)
    
    # Calculate weighted fitness
    fitness = 0.0
    total_weight = 0.0
    
    for metric_name, weight in weights.items():
        if metric_name in scores:
            fitness += scores[metric_name] * weight
            total_weight += weight
    
    # Normalize by total weight (handles missing metrics)
    if total_weight > 0:
        fitness /= total_weight
    
    # Clamp to [0, 1]
    fitness = max(0.0, min(1.0, fitness))
    
    if return_details:
        return fitness, metrics, scores
    
    return fitness


# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_all_metrics(morphology_array: np.ndarray) -> Dict[str, float]:
    """
    Calculate all geometric metrics for a morphology.
    
    Returns:
        Dictionary with metric names and values
    """
    #print("AAAAAAAAAA22")
    metrics = {}
    #print("AAAAAAAAAA23", morphology_array)   
    # Get filled voxel positions
    filled = morphology_array > 0
    #print("AAAAAAAAAA24")
    voxel_positions = np.argwhere(filled)
    #print("AAAAAAAAAA25")    
    if len(voxel_positions) == 0:
        # Empty morphology - return zero metrics
        return {key: 0.0 for key in [
            'connectivity', 'fill_ratio', 'aspect_ratio', 'compactness',
            'elongation', 'radial_symmetry', 'bilateral_symmetry',
            'spherical_symmetry', 'hollowness'
        ]}
    #print("AAAAAAAAAA26")   
    # Basic metrics
    metrics['connectivity'] = measure_connectivity(filled)
    metrics['fill_ratio'] = measure_fill_ratio(filled)
    
    # Bounding box metrics
    metrics['aspect_ratio'] = measure_aspect_ratio(voxel_positions)
    
    # Shape complexity
    metrics['compactness'] = measure_compactness(filled)
    metrics['elongation'] = measure_elongation(voxel_positions)
    
    # Symmetry metrics

    metrics['circular_symmetry'] = measure_circular_symmetry(morphology_array)

    #metrics['radial_symmetry'] = measure_radial_symmetry(morphology_array)
    metrics['bilateral_symmetry'] = measure_bilateral_symmetry(filled)
    metrics['spherical_symmetry'] = measure_spherical_symmetry(voxel_positions)
    
    # Structure metrics
    metrics['hollowness'] = measure_hollowness(morphology_array)
    
    return metrics


def measure_connectivity(filled: np.ndarray) -> float:
    """
    Check if morphology is single connected component.
    
    Returns:
        1.0 if single body, 1/N if N components, 0.0 if empty
    """
    labeled, num_components = label(filled)
    
    if num_components == 0:
        return 0.0
    elif num_components == 1:
        return 1.0
    else:
        # Penalty for multiple components
        return 1.0 / num_components


def measure_fill_ratio(filled: np.ndarray) -> float:
    """
    Calculate ratio of filled voxels to total volume.
    
    Returns:
        Fill ratio [0, 1]
    """
    total_voxels = filled.size
    filled_voxels = np.sum(filled)
    
    if total_voxels == 0:
        return 0.0
    
    return filled_voxels / total_voxels


def measure_aspect_ratio(voxel_positions: np.ndarray) -> float:
    """
    Calculate aspect ratio from bounding box.
    
    Returns:
        Length / width ratio
    """
    if len(voxel_positions) == 0:
        return 1.0
    
    mins = voxel_positions.min(axis=0)
    maxs = voxel_positions.max(axis=0)
    dimensions = maxs - mins + 1  # [z, y, x]
    
    # Length = largest dimension, width = second largest
    sorted_dims = np.sort(dimensions)
    length = sorted_dims[-1]
    width = sorted_dims[-2]
    
    if width == 0:
        return 1.0
    
    return length / width


def measure_compactness(filled: np.ndarray) -> float:
    """
    Measure compactness as volume^(2/3) / surface_area.
    Sphere = 1.0 (maximum), lower for elongated shapes.
    
    Returns:
        Compactness score [0, 1]
    """
    volume = np.sum(filled)
    
    if volume == 0:
        return 0.0
    
    # Approximate surface area (voxels with at least one empty neighbor)
    surface_voxels = 0
    z_size, y_size, x_size = filled.shape
    
    for z, y, x in np.argwhere(filled):
        # Check 6 neighbors
        neighbors = [
            (z-1, y, x), (z+1, y, x),
            (z, y-1, x), (z, y+1, x),
            (z, y, x-1), (z, y, x+1)
        ]
        
        for nz, ny, nx in neighbors:
            if (0 <= nz < z_size and 
                0 <= ny < y_size and 
                0 <= nx < x_size):
                if not filled[nz, ny, nx]:
                    surface_voxels += 1
                    break
            else:
                # Border is also surface
                surface_voxels += 1
                break
    
    if surface_voxels == 0:
        return 0.0
    
    # Ideal sphere: volume^(2/3) / surface ≈ constant
    compactness = (volume ** (2.0/3.0)) / surface_voxels
    
    # Normalize approximately (sphere ≈ 0.5 for voxelized shapes)
    normalized = compactness / 0.5
    
    return min(normalized, 1.0)


def measure_elongation(voxel_positions: np.ndarray) -> float:
    """
    Measure elongation using PCA eigenvalue ratio.
    
    Returns:
        Ratio of largest to second eigenvalue (1.0 = sphere, >1 = elongated)
    """
    if len(voxel_positions) < 3:
        return 1.0
    
    # Center the data
    centered = voxel_positions - voxel_positions.mean(axis=0)
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    
    if eigenvalues[1] == 0:
        return 1.0
    
    elongation = eigenvalues[0] / eigenvalues[1]
    
    return elongation




def measure_circular_symmetry(morphology_array: np.ndarray) -> float:
    """Measure circular symmetry across all z-slices."""
    z_size, y_size, x_size = morphology_array.shape
    slice_scores = []
    
    for z in range(z_size):
        slice_2d = morphology_array[z, :, :]
        if np.sum(slice_2d > 0) == 0:
            continue
        
        filled_positions = np.argwhere(slice_2d > 0)
        center_y = np.mean(filled_positions[:, 0])
        center_x = np.mean(filled_positions[:, 1])
        #for each slice different y center?
        
        score = measure_slice_circularity(slice_2d, center_y, center_x)
        slice_scores.append(score)
    
    return np.mean(slice_scores) if len(slice_scores) > 0 else 0.0


def measure_slice_circularity(slice_2d: np.ndarray,
                              center_y: float,
                              center_x: float) -> float:
    """Measure circularity of single slice using 8 radial rays."""

    # CROP TO BOUNDING BOX FIRST
    filled_positions = np.argwhere(slice_2d > 0)
    if len(filled_positions) == 0:
        return 0.0

    y_min, y_max = filled_positions[:, 0].min(), filled_positions[:, 0].max()
    x_min, x_max = filled_positions[:, 1].min(), filled_positions[:, 1].max()

    # Crop the slice
    cropped_slice = slice_2d[y_min:y_max + 1, x_min:x_max + 1]

    # Recalculate center relative to cropped slice
    cropped_filled = np.argwhere(cropped_slice > 0)
    center_y = np.mean(cropped_filled[:, 0])
    center_x = np.mean(cropped_filled[:, 1])

    # Now use cropped_slice instead of slice_2d
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    distances = []

    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)
        dy = np.sin(angle_rad)
        dx = np.cos(angle_rad)

        distance = 0
        max_distance = max(cropped_slice.shape)  # Changed

        for step in range(1, max_distance):
            y = int(center_y + dy * step)
            x = int(center_x + dx * step)

            if not (0 <= y < cropped_slice.shape[0] and 0 <= x < cropped_slice.shape[1]):
                # Hit grid boundary - use last filled position we saw
                break

            if cropped_slice[y, x] > 0:
                distance = step  # Keep updating as we find filled voxels
                # Don't break - keep walking to find the FURTHEST filled voxel

        # After loop ends, distance = furthest filled voxel along this ray
        if distance > 0:
            distances.append(distance)
    
    if len(distances) < 4:
        return 0.0
    #
    # cv = 0
    # print(distances)
    # for i in range (0, len(distances)-1) :
    #     d1 = distances[ i]
    #     d2 = distances[ i + 1]
    #     if ((d1/d2 > 0.66) and (d1/d2 < 1.51)) : cv = cv + 0.125
    # d1 = distances[0]
    # d2 = distances[ -1]
    # if ((d1/d2 > 0.66) and (d1/d2 < 1.51)) : cv = cv + 0.125


    distances = np.array(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    print("CCCC1", distances, mean_dist, std_dist)
    if mean_dist == 0:
        return 0.0
    
    cv = std_dist / mean_dist

    # Detect cube pattern - cardinals different from diagonals
    angles_list = [0, 45, 90, 135, 180, 225, 270, 315]
    is_bimodal = False

    if len(distances) == 8:
        cardinals = [distances[0], distances[2], distances[4], distances[6]]
        diagonals = [distances[1], distances[3], distances[5], distances[7]]

        mean_cardinal = np.mean(cardinals)
        mean_diagonal = np.mean(diagonals)

        # Check 1: Square pattern (cardinals vs diagonals)
        ratio = max(mean_cardinal, mean_diagonal) / min(mean_cardinal, mean_diagonal)

        # Check 2: Rectangle pattern (cardinals vary among themselves)
        cardinal_cv = np.std(cardinals) / mean_cardinal if mean_cardinal > 0 else 0

        print(f"DEBUG: cardinal_mean={mean_cardinal:.2f}, diagonal_mean={mean_diagonal:.2f}, ratio={ratio:.2f}")
        print(f"DEBUG: cardinal_cv={cardinal_cv:.2f}, cardinals={cardinals}")

        # Bimodal if EITHER:
        # - Square pattern (cardinal/diagonal ratio > 1.25), OR
        # - Rectangle pattern (cardinals vary by >15%)
        is_bimodal = (ratio > 1.25) or (cardinal_cv > 0.15)
    
    # Threshold scoring
    print ("cv", cv)
    if cv < 0.08:
        base_score = 1.0
    elif cv < 0.15:
        base_score = 1.0 - (cv - 0.08) / 0.07 * 0.2
    elif cv < 0.25:
        base_score = 0.8 - (cv - 0.15) / 0.10 * 0.3
    elif cv < 0.35:
        base_score = 0.5 - (cv - 0.25) / 0.10 * 0.3
    else:
        base_score = max(0.2 - (cv - 0.35) * 0.4, 0.0)
    print(base_score)    
    if is_bimodal:
        base_score *= 0.5
        print("is bimodal")
    print(base_score)
    return base_score

    print(cv)
    return cv





def measure_radial_symmetry(morphology_array: np.ndarray) -> float:
    """
    Measure radial symmetry around z-axis.
    Checks if density is uniform at each radius.
    
    Returns:
        Symmetry score [0, 1], 1.0 = perfect radial symmetry
    """
    z_size, y_size, x_size = morphology_array.shape
    
    if z_size == 0:
        return 0.0
    
    center_y = y_size / 2.0
    center_x = x_size / 2.0
    
    symmetry_scores = []
    
    for z in range(z_size):
        slice_2d = morphology_array[z, :, :]
        
        if np.sum(slice_2d > 0) == 0:
            continue
        
        # Compute radial density profile
        profile = compute_radial_profile(slice_2d, center_y, center_x)
        
        if len(profile) > 0:
            # Low variance = high symmetry
            std = np.std(profile)
            mean = np.mean(profile)
            
            if mean > 0:
                cv = std / mean  # Coefficient of variation
                symmetry_score = np.exp(-cv * 2)  # Exponential decay
                symmetry_scores.append(symmetry_score)
    
    if len(symmetry_scores) == 0:
        return 0.0
    
    return np.mean(symmetry_scores)


def compute_radial_profile(slice_2d: np.ndarray, 
                          center_y: float, 
                          center_x: float) -> np.ndarray:
    """
    Compute radial density profile for a 2D slice.
    
    Returns:
        Array of densities at each radius
    """
    y_coords, x_coords = np.meshgrid(
        range(slice_2d.shape[0]), 
        range(slice_2d.shape[1]), 
        indexing='ij'
    )
    
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    
    # Bin by distance
    max_dist = int(np.max(distances))
    profile = []
    
    for r in range(max_dist):
        mask = (distances >= r) & (distances < r+1)
        if np.sum(mask) > 0:
            density = np.mean(slice_2d[mask] > 0)
            profile.append(density)
    
    return np.array(profile)


def measure_bilateral_symmetry(filled: np.ndarray) -> float:
    """
    Measure bilateral symmetry (mirror across x-axis plane).
    
    Returns:
        Symmetry score [0, 1], 1.0 = perfect mirror symmetry
    """
    z_size, y_size, x_size = filled.shape
    
    # Mirror across x-axis (left-right)
    left_half = filled[:, :, :x_size//2]
    right_half = filled[:, :, x_size//2:]
    right_half_flipped = np.flip(right_half, axis=2)
    
    # Pad to same size
    min_width = min(left_half.shape[2], right_half_flipped.shape[2])
    left_half = left_half[:, :, :min_width]
    right_half_flipped = right_half_flipped[:, :, :min_width]
    
    # Measure similarity
    total_voxels = left_half.size
    if total_voxels == 0:
        return 0.0
    
    matching = np.sum(left_half == right_half_flipped)
    symmetry_score = matching / total_voxels
    
    return symmetry_score


def measure_spherical_symmetry(voxel_positions: np.ndarray) -> float:
    """
    Measure spherical symmetry (all points equidistant from center).
    
    Returns:
        Symmetry score [0, 1], 1.0 = perfect sphere
    """
    if len(voxel_positions) == 0:
        return 0.0
    
    # Center point
    center = voxel_positions.mean(axis=0)
    
    # Distances from center
    distances = np.linalg.norm(voxel_positions - center, axis=1)
    
    # Spherical = low variance in distances
    std = np.std(distances)
    mean = np.mean(distances)
    
    if mean == 0:
        return 0.0
    
    cv = std / mean  # Coefficient of variation
    symmetry_score = np.exp(-cv * 2)  # Exponential decay
    
    return symmetry_score


def measure_hollowness(morphology_array: np.ndarray) -> float:
    """
    Measure hollowness (empty center, filled edges).
    
    Returns:
        Hollowness score [0, 1], 1.0 = perfect hollow cylinder
    """
    z_size, y_size, x_size = morphology_array.shape
    
    hollow_scores = []
    
    for z in range(z_size):
        slice_2d = morphology_array[z, :, :]
        
        # Define center region (middle 40% of slice)
        center_y_start = int(y_size * 0.3)
        center_y_end = int(y_size * 0.7)
        center_x_start = int(x_size * 0.3)
        center_x_end = int(x_size * 0.7)
        
        center_region = slice_2d[center_y_start:center_y_end, 
                                 center_x_start:center_x_end]
        
        # Edge region (everything except center)
        edge_region = slice_2d.copy()
        edge_region[center_y_start:center_y_end, 
                   center_x_start:center_x_end] = 0
        
        if center_region.size == 0 or edge_region.size == 0:
            continue
        
        # Hollow = empty center + filled edges
        center_empty = np.sum(center_region == 0) / center_region.size
        edge_filled = np.sum(edge_region > 0) / edge_region.size
        
        hollow_score = center_empty * edge_filled
        hollow_scores.append(hollow_score)
    
    if len(hollow_scores) == 0:
        return 0.0
    
    return np.mean(hollow_scores)


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def check_hard_constraints(metrics: Dict[str, float]) -> bool:
    """
    Check if morphology passes hard constraints (fatal failures).
    
    Returns:
        True if passes, False if fails
    """
    # Must be single connected body
    if metrics.get('connectivity', 0.0) < 0.5:
        return False
    
    # Must not be empty or solid blob
    fill_ratio = metrics.get('fill_ratio', 0.0)
    if fill_ratio < 0.05 or fill_ratio > 0.95:
        return False
    
    return True


def score_metrics(metrics: Dict[str, float], 
                 targets: Dict[str, any],
                 target_shape: str) -> Dict[str, float]:
    """
    Convert raw metric values to normalized scores [0, 1].
    
    Args:
        metrics: Raw metric values
        targets: Target values for this shape
        target_shape: Shape name (for context)
    
    Returns:
        Dictionary of scores [0, 1] for each metric
    """
    scores = {}
    
    # Connectivity (already [0, 1])
    scores['connectivity'] = metrics.get('connectivity', 0.0)
    
    # Fill ratio (already [0, 1], ideal 0.2-0.8)
    fill = metrics.get('fill_ratio', 0.0)
    if 0.2 <= fill <= 0.8:
        scores['fill_ratio'] = 0.1 #1.0
    elif fill < 0.2:
        scores['fill_ratio'] = fill / 0.2
    else:
        scores['fill_ratio'] = (1.0 - fill) / 0.2
    
    # Aspect ratio (score based on target range)
    aspect = metrics.get('aspect_ratio', 1.0)
    if 'aspect_ratio' in targets:
        target = targets['aspect_ratio']
        if isinstance(target, tuple):
            # Range target
            min_val, max_val = target
            if min_val <= aspect <= max_val:
                scores['aspect_ratio'] = 1.0
            else:
                # Exponential decay outside range
                if aspect < min_val:
                    error = (min_val - aspect) / min_val
                else:
                    error = (aspect - max_val) / max_val
                scores['aspect_ratio'] = np.exp(-error * 2)
        else:
            # Point target
            error = abs(aspect - target) / max(target, 1.0)
            scores['aspect_ratio'] = np.exp(-error * 2)
    else:
        scores['aspect_ratio'] = 0.5  # Neutral
    
    # Compactness (score based on target)
    if 'compactness' in targets:
        compact = metrics.get('compactness', 0.0)
        target = targets['compactness']
        error = abs(compact - target) / target
        scores['compactness'] = np.exp(-error * 2)
    
    # Elongation (score based on target)
    if 'elongation' in targets:
        elong = metrics.get('elongation', 1.0)
        target = targets['elongation']
        error = abs(elong - target) / target
        scores['elongation'] = np.exp(-error * 2)
    
    # Symmetry scores (already [0, 1], score based on target)
    for sym_type in ['radial_symmetry', 'circular_symmetry', 'bilateral_symmetry', 'spherical_symmetry']:
        if sym_type in targets:
            sym_value = metrics.get(sym_type, 0.0)
            target = targets[sym_type]
            error = abs(sym_value - target)
            scores[sym_type] = np.exp(-error * 3)  # Strict symmetry requirement
        else:
            scores[sym_type] = metrics.get(sym_type, 0.0)
    
    # Hollowness (score based on target)
    if 'hollowness' in targets:
        hollow = metrics.get('hollowness', 0.0)
        target = targets['hollowness']
        error = abs(hollow - target)
        scores['hollowness'] = np.exp(-error * 2)
    
    return scores


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def morphology_to_array(morphology: list, dimensions: dict) -> np.ndarray:
    """
    Convert morphology string list to 3D numpy array.
    
    Args:
        morphology: List of layer strings (from DesignEngine)
        dimensions: Dict with 'x', 'y', 'z' keys
    
    Returns:
        3D numpy array (Z, Y, X)
    """
    z_size = len(morphology)
    y_size = dimensions['y']
    x_size = dimensions['x']
    
    array = np.zeros((z_size, y_size, x_size), dtype=int)
    
    for z, layer in enumerate(morphology):
        for idx, char in enumerate(layer):
            y = idx // x_size
            x = idx % x_size
            if y < y_size and x < x_size:
                array[z, y, x] = int(char)
    
    return array


def print_fitness_report(morphology_array: np.ndarray, 
                        target_shape: str,
                        genome_id: int = None):
    """
    Print detailed fitness report for debugging.
    
    Args:
        morphology_array: Morphology to evaluate
        target_shape: Target shape name
        genome_id: Optional genome ID for labeling
    """
    fitness, metrics, scores = calculate_fitness(
        morphology_array, target_shape, return_details=True
    )
    
    print("\n" + "="*60)
    if genome_id is not None:
        print(f"FITNESS REPORT - Genome {genome_id} ({target_shape})")
    else:
        print(f"FITNESS REPORT ({target_shape})")
    print("="*60)
    
    print(f"\nFinal Fitness: {fitness:.3f}")
    print(f"Shape dimensions: {morphology_array.shape}")
    
    print("\n--- Raw Metrics ---")
    for name, value in sorted(metrics.items()):
        print(f"  {name:25s}: {value:.3f}")
    
    print("\n--- Normalized Scores [0,1] ---")
    for name, value in sorted(scores.items()):
        print(f"  {name:25s}: {value:.3f}")
    
    # Show which metrics are weighted for this shape
    weights = SHAPE_WEIGHT_PRESETS[target_shape]
    print("\n--- Active Weights ---")
    for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        score = scores.get(name, 0.0)
        contribution = score * weight
        print(f"  {name:25s}: weight={weight:.2f}, score={score:.3f}, contrib={contribution:.3f}")
    
    print("="*60 + "\n")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Geometric Fitness Module - Testing")
    print("="*60)
    
    # Create test morphologies
    print("\n1. Testing with synthetic morphologies...")
    
    # Test 1: Elongated cylinder-like shape
    cylinder_morph = np.zeros((60, 8, 15), dtype=int)
    for z in range(60):
        for y in range(2, 6):
            for x in range(5, 10):
                # Hollow cylinder
                if not (3 <= y <= 4 and 6 <= x <= 8):
                    cylinder_morph[z, y, x] = 1
    
    print("\nTest: Cylinder-like morphology")
    fitness = calculate_fitness(cylinder_morph, 'cylinder')
    print(f"Fitness: {fitness:.3f}")
    print_fitness_report(cylinder_morph, 'cylinder')
    
    # Test 2: Sphere-like shape
    sphere_morph = np.zeros((20, 20, 20), dtype=int)
    center = np.array([10, 10, 10])
    for z in range(20):
        for y in range(20):
            for x in range(20):
                dist = np.linalg.norm(np.array([z, y, x]) - center)
                if 5 < dist < 8:
                    sphere_morph[z, y, x] = 1
    
    print("\nTest: Sphere-like morphology")
    fitness = calculate_fitness(sphere_morph, 'sphere')
    print(f"Fitness: {fitness:.3f}")
    
    # Test 3: Random blob (should score low)
    random_morph = np.random.randint(0, 2, (10, 10, 10))
    
    print("\nTest: Random blob")
    fitness = calculate_fitness(random_morph, 'cylinder')
    print(f"Fitness: {fitness:.3f}")
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("\nAvailable target shapes:", list(SHAPE_WEIGHT_PRESETS.keys()))
