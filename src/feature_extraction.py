#!/usr/bin/env python3
"""
Feature Extraction Module for BHM Ontology-to-CPPN Mapping

Transforms ontology instances into categorical morphological features
that can be mapped to CPPN activation function weights.

Author: BioMeld Project
Date: 2025-11-14 (Updated for Master Ontology v4.0 compliance)
"""

import json
from typing import Dict, Any, Optional


def extract_morphology_features(ontology_instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract categorical morphological features from ontology instance.
    
    Args:
        ontology_instance: Dictionary containing problem specification with
                          'geometric' and 'functional' sections
                          
    Returns:
        Dictionary of categorical features including:
        - base_morphology (MANDATORY in v4.0)
        - symmetry_type, symmetry_axes_count
        - aspect_ratio_category, size_category
        - structural features (segmented, branched, hollow)
        - functional features (locomotion, actuation)
    
    Example:
        >>> instance = load_json('blood_vessel_catheter.json')
        >>> features = extract_morphology_features(instance)
        >>> print(features['symmetry_type'])
        'bilateral'
    """
    geometric = ontology_instance.get('geometric', {})
    functional = ontology_instance.get('functional', {})
    function_weights = ontology_instance.get('function_weights', {})
    
    features = {
        'function_weights': function_weights,
        # === MORPHOLOGY ===
        'base_morphology': geometric.get('base_morphology', 'irregular'),
        
        # === SYMMETRY ===
        'symmetry_type': geometric.get('symmetry_type', 'asymmetric'),
        'symmetry_axes_count': len(geometric.get('symmetry_axes', [])),
        
        # === DIMENSIONAL CATEGORIES ===
        'aspect_ratio_category': classify_aspect_ratio(
            geometric.get('aspect_ratio', 1.0)
        ),
        'size_category': classify_size(
            geometric.get('max_length', {}).get('value', 10),
            geometric.get('max_diameter', {}).get('value', 10)
        ),
        
        # === STRUCTURE ===
        'connectivity_type': geometric.get('connectivity_type', 'single_body'),
        'is_segmented': geometric.get('is_segmented', False),
        'segment_count': geometric.get('segment_count', 0),
        'segment_uniformity': geometric.get('segment_uniformity', 'identical'),
        
        # === HOLLOWNESS ===
        'hollowness_type': geometric.get('hollowness_type', 'solid'),
        'has_internal_channels': geometric.get('has_internal_channels', False),
        
        # === APPENDAGES & BRANCHING ===
        'has_appendages': geometric.get('surface_type') == 'with_appendages',
        'appendage_count': geometric.get('appendage_count', 0),
        'is_branched': geometric.get('connectivity_type') == 'branched',
        
        # === GRADIENTS ===
        'needs_spatial_gradient': detect_gradient_needs(ontology_instance),
        
        # === FUNCTIONAL ===
        'primary_function': functional.get('primary_function', 'unknown'),
        'locomotion_type': functional.get('locomotion_type'),
        'actuation_pattern': functional.get('actuation_pattern'),
        'is_confined': functional.get('confined_tubular', False),
    }
    
    return features


def classify_aspect_ratio(ratio: float) -> str:
    """
    Convert continuous aspect ratio to categorical feature.
    
    Args:
        ratio: Aspect ratio (length:width, dimensionless)
        
    Returns:
        Category string: 'very_elongated', 'elongated', 'moderate', or 'compact'
    """
    if ratio > 15:
        return 'very_elongated'
    elif ratio > 8:
        return 'elongated'
    elif ratio > 3:
        return 'moderate'
    else:
        return 'compact'


def classify_size(length_mm: float, diameter_mm: float) -> str:
    """
    Categorize by approximate volume scale.
    
    Args:
        length_mm: Maximum length in millimeters
        diameter_mm: Maximum diameter in millimeters
        
    Returns:
        Size category: 'micro', 'small', 'medium', or 'large'
    """
    volume_mm3 = length_mm * diameter_mm ** 2  # Rough approximation
    
    if volume_mm3 < 10:
        return 'micro'
    elif volume_mm3 < 1000:
        return 'small'
    elif volume_mm3 < 100000:
        return 'medium'
    else:
        return 'large'


def detect_gradient_needs(ontology_instance: Dict[str, Any]) -> bool:
    """
    Determine if spatial gradients are required based on multiple indicators.
    
    Args:
        ontology_instance: Full problem specification
        
    Returns:
        True if gradients needed, False otherwise
    """
    geometric = ontology_instance.get('geometric', {})
    functional = ontology_instance.get('functional', {})
    
    indicators = [
        geometric.get('segment_uniformity') == 'gradient',
        geometric.get('hollowness_type') == 'partially_hollow',
        functional.get('requires_spatial_gradient', False),
    ]
    
    return any(indicators)


def load_ontology_instance(filepath: str) -> Dict[str, Any]:
    """
    Load ontology instance from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary containing ontology instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_features(features: Dict[str, Any], filepath: str) -> None:
    """
    Save extracted features to JSON file.
    
    Args:
        features: Dictionary of extracted features
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(features, f, indent=2)


# Example usage
if __name__ == "__main__":
    # Example: Extract features from test case
    example_instance = {
        "problem_id": "example_catheter",
        "geometric": {
            "max_length": {"value": 60, "unit": "mm"},
            "max_diameter": {"value": 3, "unit": "mm"},
            "aspect_ratio": 20,
            "base_morphology": "elongated_cylinder",
            "symmetry_type": "bilateral",
            "symmetry_axes": ["x"],
            "connectivity_type": "single_body",
            "is_segmented": False,
            "hollowness_type": "hollow",
            "has_internal_channels": True,
            "surface_type": "smooth"
        },
        "functional": {
            "primary_function": "navigation",
            "locomotion_type": "passive",
            "confined_tubular": True
        }
    }
    
    features = extract_morphology_features(example_instance)
    
    print("Extracted Features:")
    print(json.dumps(features, indent=2))
    
    # Expected output:
    # {
    #   "base_morphology": "elongated_cylinder",
    #   "symmetry_type": "bilateral",
    #   "aspect_ratio_category": "very_elongated",
    #   "hollowness_type": "hollow",
    #   ...
    # }
