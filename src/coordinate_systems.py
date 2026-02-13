"""
Coordinate system transformations for CPPN morphology generation.
Supports Cartesian, Cylindrical, and Spherical coordinate systems.

Author: BioMeld Project
Date: December 2025
Version: 1.0
"""

import numpy as np
from typing import Tuple, Dict, Any


class CoordinateTransformer:
    """Handles coordinate transformations and normalization for CPPNs."""
    
    SUPPORTED_SYSTEMS = ['cartesian', 'cylindrical', 'spherical']
    
    def __init__(self, coordinate_system: str, dimensions: Dict[str, int]):
        """
        Initialize coordinate transformer.
        
        Args:
            coordinate_system: 'cartesian', 'cylindrical', or 'spherical'
            dimensions: {'x': int, 'y': int, 'z': int} - Cartesian grid size
        
        Raises:
            ValueError: If coordinate_system not supported
        """
        if coordinate_system not in self.SUPPORTED_SYSTEMS:
            raise ValueError(
                f"Unsupported coordinate system: {coordinate_system}. "
                f"Must be one of {self.SUPPORTED_SYSTEMS}"
            )
        
        self.system = coordinate_system
        self.dimensions = dimensions
        
        # Calculate center point (for cylindrical/spherical)
        self.center_x = dimensions['x'] / 2.0
        self.center_y = dimensions['y'] / 2.0
        self.center_z = dimensions['z'] / 2.0
        
        # Calculate coordinate ranges for normalization
        self._calculate_ranges()
    
    def _calculate_ranges(self):
        """Calculate min/max ranges for each coordinate in target system."""
        if self.system == 'cartesian':
            self.ranges = {
                'x': (0, self.dimensions['x']),
                'y': (0, self.dimensions['y']),
                'z': (0, self.dimensions['z'])
            }
        
        elif self.system == 'cylindrical':
            # Maximum radius from center to corner
            max_r = np.sqrt(self.center_x**2 + self.center_y**2)
            self.ranges = {
                'r': (0, max_r),
                'theta': (-np.pi, np.pi),
                'z': (0, self.dimensions['z'])
            }
        
        elif self.system == 'spherical':
            # Maximum radius from center to corner
            max_r = np.sqrt(self.center_x**2 + self.center_y**2 + self.center_z**2)
            self.ranges = {
                'r': (0, max_r),
                'theta': (-np.pi, np.pi),
                'phi': (0, np.pi)
            }
    
    def transform(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Transform Cartesian coordinates to target system.
        
        Args:
            x, y, z: Cartesian coordinates from voxel grid
            
        Returns:
            Tuple of coordinates in target system:
                - Cartesian: (x, y, z)
                - Cylindrical: (r, theta, z)
                - Spherical: (r, theta, phi)
        """
        if self.system == 'cartesian':
            return (x, y, z)
        
        elif self.system == 'cylindrical':
            # Shift origin to center for x and y
            dx = x - self.center_x
            dy = y - self.center_y
            
            # Calculate cylindrical coordinates
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            # z stays the same
            return (r, theta, z)
        
        elif self.system == 'spherical':
            # Shift origin to center
            dx = x - self.center_x
            dy = y - self.center_y
            dz = z - self.center_z
            
            # Calculate spherical coordinates
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            theta = np.arctan2(dy, dx)  # Azimuthal angle
            phi = np.arccos(dz / (r + 1e-10))  # Polar angle (avoid division by zero)
            return (r, theta, phi)
    
    def normalize(self, coords: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Normalize coordinates to [-1, 1] range for CPPN input.
        
        CPPNs typically expect inputs in [-1, 1] range for optimal learning.
        
        Args:
            coords: Coordinates in target system (from transform())
            
        Returns:
            Normalized coordinates in [-1, 1] range
        """
        if self.system == 'cartesian':
            x, y, z = coords
            x_norm = (2 * x / self.dimensions['x']) - 1
            y_norm = (2 * y / self.dimensions['y']) - 1
            z_norm = (2 * z / self.dimensions['z']) - 1
            return (x_norm, y_norm, z_norm)
        
        elif self.system == 'cylindrical':
            r, theta, z = coords
            # r: [0, max_r] → [-1, 1]
            r_norm = (2 * r / self.ranges['r'][1]) - 1
            # theta: [-π, π] → [-1, 1]
            theta_norm = theta / np.pi
            # z: [0, max_z] → [-1, 1]
            z_norm = (2 * z / self.dimensions['z']) - 1
            return (r_norm, theta_norm, z_norm)
        
        elif self.system == 'spherical':
            r, theta, phi = coords
            # r: [0, max_r] → [-1, 1]
            r_norm = (2 * r / self.ranges['r'][1]) - 1
            # theta: [-π, π] → [-1, 1]
            theta_norm = theta / np.pi
            # phi: [0, π] → [-1, 1]
            phi_norm = (2 * phi / np.pi) - 1
            return (r_norm, theta_norm, phi_norm)
    
    def transform_and_normalize(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convenience method: transform then normalize in one call.
        
        Args:
            x, y, z: Cartesian coordinates from voxel grid
            
        Returns:
            Normalized coordinates in target system (ready for CPPN input)
        """
        transformed = self.transform(x, y, z)
        return self.normalize(transformed)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about current coordinate system configuration.
        
        Returns:
            Dictionary with system info and ranges
        """
        return {
            'coordinate_system': self.system,
            'dimensions': self.dimensions,
            'center': {
                'x': self.center_x,
                'y': self.center_y,
                'z': self.center_z
            },
            'ranges': self.ranges
        }


def select_coordinate_system_from_ontology(parameters_data: Dict[str, Any]) -> str:
    """
    Auto-select coordinate system based on ontology parameters.
    
    Selection priority:
    1. Explicit specification ('coordinate_system' parameter)
    2. Archetype-based selection
    3. Symmetry-based selection
    4. Base morphology
    5. Default to Cartesian
    
    Args:
        parameters_data: Dictionary with ontology parameters
        
    Returns:
        Coordinate system name: 'cartesian', 'cylindrical', or 'spherical'
    
    Example:
        >>> params = {'archetype': 1, 'symmetry_type': 'radial'}
        >>> system = select_coordinate_system_from_ontology(params)
        >>> print(system)
        'cylindrical'
    """
    # Priority 1: Explicit specification
    if 'coordinate_system' in parameters_data:
        return parameters_data['coordinate_system']
    
    # Priority 2: Archetype-based selection
    archetype = parameters_data.get('archetype', None)
    if archetype in [1, 2, 5]:  # Elongated, Compact, Segmented
        return 'cylindrical'
    elif archetype == 3:  # Sphere/Ellipsoid
        return 'spherical'
    elif archetype in [4, 6]:  # Branched, Planar
        return 'cartesian'
    
    # Priority 3: Symmetry-based selection
    symmetry = parameters_data.get('symmetry_type', 'asymmetric')
    if symmetry in ['radial', 'axial_cylindrical']:
        return 'cylindrical'
    elif symmetry == 'spherical':
        return 'spherical'
    elif symmetry in ['bilateral', 'planar']:
        return 'cartesian'
    
    # Priority 4: Base morphology
    base_morph = parameters_data.get('base_morphology', '')
    if 'cylinder' in base_morph or 'elongated' in base_morph:
        return 'cylindrical'
    elif 'sphere' in base_morph or 'ellipsoid' in base_morph:
        return 'spherical'
    elif 'branched' in base_morph or 'planar' in base_morph:
        return 'cartesian'
    
    # Default: Cartesian (most flexible)
    return 'cartesian'


# Testing utilities
def test_coordinate_transformation():
    """Test coordinate transformations with known values."""
    print("Testing Coordinate Transformations")
    print("=" * 60)
    
    dimensions = {'x': 10, 'y': 10, 'z': 20}
    
    # Test Cartesian (identity)
    print("\n1. Cartesian (Identity Transform):")
    cart_transformer = CoordinateTransformer('cartesian', dimensions)
    x, y, z = 5, 5, 10
    transformed = cart_transformer.transform(x, y, z)
    normalized = cart_transformer.normalize(transformed)
    print(f"   Input: ({x}, {y}, {z})")
    print(f"   Transformed: {transformed}")
    print(f"   Normalized: {normalized}")
    
    # Test Cylindrical
    print("\n2. Cylindrical:")
    cyl_transformer = CoordinateTransformer('cylindrical', dimensions)
    x, y, z = 8, 5, 10  # Off-center point
    transformed = cyl_transformer.transform(x, y, z)
    normalized = cyl_transformer.normalize(transformed)
    print(f"   Input: ({x}, {y}, {z})")
    print(f"   Transformed (r, θ, z): ({transformed[0]:.2f}, {transformed[1]:.2f}, {transformed[2]:.2f})")
    print(f"   Normalized: ({normalized[0]:.2f}, {normalized[1]:.2f}, {normalized[2]:.2f})")
    
    # Test Spherical
    print("\n3. Spherical:")
    sph_transformer = CoordinateTransformer('spherical', dimensions)
    x, y, z = 8, 5, 15
    transformed = sph_transformer.transform(x, y, z)
    normalized = sph_transformer.normalize(transformed)
    print(f"   Input: ({x}, {y}, {z})")
    print(f"   Transformed (r, θ, φ): ({transformed[0]:.2f}, {transformed[1]:.2f}, {transformed[2]:.2f})")
    print(f"   Normalized: ({normalized[0]:.2f}, {normalized[1]:.2f}, {normalized[2]:.2f})")
    
    print("\n" + "=" * 60)
    print("✅ All transformations completed successfully!")


if __name__ == "__main__":
    # Run tests
    test_coordinate_transformation()
    
    # Test auto-selection
    print("\n\nTesting Auto-Selection:")
    print("=" * 60)
    
    test_cases = [
        {'archetype': 1, 'symmetry_type': 'radial'},
        {'archetype': 3, 'base_morphology': 'sphere'},
        {'archetype': 4, 'symmetry_type': 'bilateral'},
        {'symmetry_type': 'axial_cylindrical'},
        {'base_morphology': 'branched'},
    ]
    
    for i, params in enumerate(test_cases, 1):
        system = select_coordinate_system_from_ontology(params)
        print(f"{i}. Params: {params}")
        print(f"   → Selected: {system}\n")
