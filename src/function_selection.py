#!/usr/bin/env python3
"""
Function Selection Module for CPPN Activation Functions

Maps morphological features to CPPN activation function weights,
then selects functions based on various strategies.

Author: BioMeld Project
Date: 2025-11-11
"""

import json
from typing import Dict, List, Any
from src.feature_extraction import extract_morphology_features





"""
 ['identity', 'sigmoid', 'tanh', 'abs', 'neg_abs', 'sqrt_abs', 'neg_sqrt_abs', 'neg_square']
PPPPPPPPPPPP3 {'functions': {
'abs':'clamped':'cube':  'elu':'exp': 'gauss''hat':'identity': 'inv': 'lelu':'log':'neg_abs':'neg_sin': 'neg_square':'neg_sqrt_abs': 'relu':'selu':'sigmoid':'sin':'softplus':'square': 'sqrt_abs':'tanh':  



"""





# Scoring matrix: feature → function weights
FUNCTION_SCORES = {
    'bilateral_symmetry': {
        'abs': 0.95,
        'neg_abs': 0.80,
        'sqrt_abs': 0.70,
        'sigmoid': 0.60,
        'tanh': 0.60,
        'gauss': 0.40,
    },
    
    'radial_symmetry': {
        'sin': 0.95,
        'neg_sin': 0.85,
        'gauss': 0.90,
        'hat': 0.60,
        'abs': 0.50,
    },
    
    'regular_segmentation': {
        'sin': 0.95,
        'neg_sin': 0.85,
        'hat': 0.60,
        'square': 0.50,
    },
    
    'smooth_gradient': {
        'sigmoid': 0.95,
        'tanh': 0.95,
        'softplus': 0.70,
        'eLU': 0.60,
        'identity': 0.80,
    },
    
    'hollow_core': {
        'neg_abs': 0.95,
        'neg_sqrt_abs': 0.90,
        'neg_square': 0.70,
    },
    
    'solid_structure': {
        'abs': 0.60,
        'sqrt_abs': 0.60,
        'identity': 0.70,
        'sigmoid': 0.50,
    },
    
    'discrete_features': {
        'relu': 0.80,
        'gauss': 0.90,
        'hat': 0.70,
        'square': 0.60,
        'clamped': 0.50,
    },
    
    'undulatory_motion': {
        'sin': 0.95,
        'neg_sin': 0.85,
        'tanh': 0.60,
    },
    
    'peristaltic_motion': {
        'sin': 0.90,
        'neg_sin': 0.80,
        'sigmoid': 0.70,
        'square': 0.60,
    },
}

# Universal functions - always somewhat useful
UNIVERSAL_FUNCTIONS = {
    'identity': 0.50,
    'sigmoid': 0.40,
    'tanh': 0.40,
}

# All available CPPN functions
ALL_FUNCTIONS = [
    'sin', 'neg_sin', 'abs', 'neg_abs',
    'square', 'neg_square', 'sqrt_abs', 
    'neg_sqrt_abs', 'sigmoid', 'clamped', 'cubical',
    'exponential', 'gauss', 'hat', 'identity', 'inverse', 'logarithmic',
    'relu', 'SeLU', 'LeLU', 'eLU', 'softplus', 'tanh'
]


def compute_function_weights(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute function weights based on extracted morphological features.
    
    Args:
        features: Dictionary from extract_morphology_features()
        
    Returns:
        Dictionary mapping function names to weight scores (0-1)
    
    Example:
        >>> features = {'symmetry_type': 'bilateral', 'hollowness_type': 'hollow'}
        >>> weights = compute_function_weights(features)
        >>> weights['abs']  # Should be high (~0.95)
    """
    # Initialize with universal baseline
    function_weights = {func: score for func, score in UNIVERSAL_FUNCTIONS.items()}
    
    # Add scores based on detected features
    if features['function_weights'] :
        _add_scores(function_weights, features['function_weights'])

    if features['symmetry_type'] == 'bilateral':
        _add_scores(function_weights, FUNCTION_SCORES['bilateral_symmetry'])
    
    if features['symmetry_type'] == 'radial':
        _add_scores(function_weights, FUNCTION_SCORES['radial_symmetry'])
    
    if features['is_segmented'] and features['segment_count'] > 3:
        _add_scores(function_weights, FUNCTION_SCORES['regular_segmentation'])
    
    if features['needs_spatial_gradient']:
        _add_scores(function_weights, FUNCTION_SCORES['smooth_gradient'])
    
    if features['hollowness_type'] == 'hollow':
        _add_scores(function_weights, FUNCTION_SCORES['hollow_core'])
    
    if features['hollowness_type'] == 'solid':
        _add_scores(function_weights, FUNCTION_SCORES['solid_structure'])
    
    if features['has_appendages'] or features['is_branched']:
        _add_scores(function_weights, FUNCTION_SCORES['discrete_features'])
    
    if features['locomotion_type'] == 'undulatory':
        _add_scores(function_weights, FUNCTION_SCORES['undulatory_motion'])
    
    if features['locomotion_type'] == 'peristaltic':
        _add_scores(function_weights, FUNCTION_SCORES['peristaltic_motion'])
    
    # Normalize to probabilities
    total = sum(function_weights.values())
    if total > 0:
        function_weights = {func: weight/total for func, weight in function_weights.items()}
    
    return function_weights


def _add_scores(weights: Dict[str, float], new_scores: Dict[str, float]) -> None:
    """Helper to add scores, taking maximum if function already exists."""
    for func, score in new_scores.items():
        weights[func] = max(weights.get(func, 0), score)


def select_functions_threshold(weights: Dict[str, float], threshold: float = 0.05) -> List[str]:
    """
    Strategy 1: Hard threshold - include only functions above threshold.
    
    Args:
        weights: Function weights from compute_function_weights()
        threshold: Minimum weight to include (default 0.05)
        
    Returns:
        List of selected function names
    """
    return [func for func, weight in weights.items() if weight >= threshold]


def select_functions_probabilistic(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Strategy 2: Probability-Weighted Sampling - return weights for probabilistic sampling.
    
    Don't exclude any functions, just return weights to be used as sampling probabilities
    during CPPN genome initialization. High-weight functions appear more often.
    
    Args:
        weights: Function weights from compute_function_weights()
        
    Returns:
        Dictionary of function weights (unchanged) to use as sampling probabilities
        
    Usage:
        weights = select_functions_probabilistic(function_weights)
        # During CPPN initialization:
        # selected = random.choices(list(weights.keys()), weights=list(weights.values()), k=genome_size)
    """
    return weights  # Return unchanged for use as sampling probabilities


def select_functions_tiered(weights: Dict[str, float]) -> Dict[str, List[str]]:
    """
    Strategy 3: Tiered selection - categorize into must-have, nice-to-have, optional.
    
    Args:
        weights: Function weights from compute_function_weights()
        
    Returns:
        Dictionary with keys 'core', 'extended', 'full'
    """
    must_have = [f for f, w in weights.items() if w >= 0.07]
    nice_to_have = [f for f, w in weights.items() if 0.04 <= w < 0.07]
    optional = [f for f, w in weights.items() if 0.015 <= w < 0.04]
    
    return {
        'core': must_have,
        'extended': must_have + nice_to_have,
        'full': must_have + nice_to_have + optional
    }


def select_functions_staged_reduction(generation: int, weights: Dict[str, float]) -> List[str]:
    """
    Strategy 4: Staged Reduction (Evolution) - start large, reduce gradually.
    
    Start with LARGE function set for broad exploration early.
    Gradually REDUCE to smaller sets for focused refinement late.
    Schedule-based (proactive, not reactive to performance).
    
    Args:
        generation: Current generation number
        weights: Function weights from compute_function_weights() (NORMALIZED)
        
    Returns:
        List of selected function names for this generation
        
    Logic:
        Gen 0-100:   Large set (T=0.02)  → 12-15 functions (keep most)
        Gen 101-300: Medium set (T=0.04) → 8-10 functions (keep moderate+)
        Gen 301+:    Small set (T=0.06)  → 4-6 functions (keep only high)
        
    Note: Thresholds are for NORMALIZED weights (sum to ~1.0)
    """
    if generation < 100:
        threshold = 0.02  # Large set - broad exploration
    elif generation < 300:
        threshold = 0.04  # Medium set - converging
    else:
        threshold = 0.06  # Small set - fine-tuning
    
    return select_functions_threshold(weights, threshold)


def select_functions_adaptive_expansion(generation: int, 
                                       weights: Dict[str, float],
                                       fitness_history: List[float],
                                       stagnation_window: int = 20) -> List[str]:
    """
    Strategy 5: Adaptive Expansion (Evolution) - start small, expand when stagnating.
    
    Start with SMALL function set for efficient search.
    EXPAND reactively when fitness stagnates (adaptive, not schedule-based).
    
    Args:
        generation: Current generation number
        weights: Function weights from compute_function_weights() (NORMALIZED)
        fitness_history: List of best fitness values over generations
        stagnation_window: Number of generations to check for stagnation (default 20)
        
    Returns:
        List of selected function names for this generation
        
    Logic:
        Gen 0-100 (not stagnating):   Small set (T=0.06)  → 4-6 functions
        Gen 101-300 (or stagnating):  Medium set (T=0.04) → 8-10 functions
        Gen 301+ (or still stagnating): Large set (T=0.02) → 12-15 functions
        
    Note: Thresholds are for NORMALIZED weights (sum to ~1.0)
    """
    # Check if stagnating
    is_stagnating = False
    if len(fitness_history) >= stagnation_window:
        recent_best = max(fitness_history[-stagnation_window:])
        previous_best = max(fitness_history[:-stagnation_window]) if len(fitness_history) > stagnation_window else 0
        
        if previous_best != 0:
            improvement = (recent_best - previous_best) / abs(previous_best)
            is_stagnating = improvement < 0.01  # Less than 1% improvement
    
    # Select threshold based on generation and stagnation
    if generation < 100 and not is_stagnating:
        threshold = 0.06  # Small set - efficient search
    elif generation < 300 and not is_stagnating:
        threshold = 0.04  # Medium set - expanded when needed
    else:
        threshold = 0.02  # Large set - full exploration when stuck
    
    return select_functions_threshold(weights, threshold)


def select_functions_for_pipeline_stage(weights: Dict[str, float], 
                                       validation_stage: str) -> List[str]:
    """
    Pipeline-Based Selection - different sets for different validation workflow stages.
    
    This is separate from evolution strategies (Strategies 4-5).
    Used across validation pipeline: visual → physics → evolution
    
    Args:
        weights: Function weights from compute_function_weights()
        validation_stage: One of 'visual_inspection', 'physics_simulation', 'evolution'
        
    Returns:
        List of selected function names
        
    Logic:
        Visual: Aggressive (T=0.06) for fast generation
        Physics: Moderate (T=0.04) to catch edge cases
        Evolution: Conservative (T=0.02) for full exploration
    """
    thresholds = {
        'visual_inspection': 0.06,  # Aggressive: ~5-7 functions
        'physics_simulation': 0.04,  # Moderate: ~10-12 functions
        'evolution': 0.02,           # Conservative: ~15-18 functions
    }
    
    threshold = thresholds.get(validation_stage, 0.01)
    return select_functions_threshold(weights, threshold)


def generate_function_report(ontology_instance: Dict[str, Any],
                            output_path: str = None) -> Dict[str, Any]:
    """
    Complete pipeline: ontology instance → features → weights → selections.
    
    Args:
        ontology_instance: Problem specification
        output_path: Optional path to save JSON report
        
    Returns:
        Complete report with features, weights, and recommended function sets
    """
    # Step 1: Extract features
    features = extract_morphology_features(ontology_instance)
    
    # Step 2: Compute weights
    weights = compute_function_weights(features)
    
    # Step 3: Generate selections for different strategies
    threshold_50 = select_functions_threshold(weights, 0.05)
    threshold_60 = select_functions_threshold(weights, 0.06)
    threshold_70 = select_functions_threshold(weights, 0.07)
    tiered = select_functions_tiered(weights)
    
    # Pipeline-based selections
    pipeline_visual = select_functions_for_pipeline_stage(weights, 'visual_inspection')
    pipeline_physics = select_functions_for_pipeline_stage(weights, 'physics_simulation')
    pipeline_evolution = select_functions_for_pipeline_stage(weights, 'evolution')
    
    # Evolution-based selections (for demonstration - requires generation/fitness data)
    evolution_staged_gen50 = select_functions_staged_reduction(50, weights)
    evolution_staged_gen200 = select_functions_staged_reduction(200, weights)
    evolution_adaptive_gen50 = select_functions_adaptive_expansion(50, weights, [])  # Empty fitness history
    
    # Compile report
    report = {
        'problem_id': ontology_instance.get('problem_id', 'unknown'),
        'problem_class': ontology_instance.get('problem_class', 'unknown'),
        'extracted_features': features,
        'function_weights': weights,
        'recommendations': {
            'aggressive': threshold_70,
            'moderate': threshold_60,
            'conservative': threshold_50,
            'tiered_core': tiered['core'],
            'tiered_extended': tiered['extended'],
            'pipeline_visual_inspection': pipeline_visual,
            'pipeline_physics_simulation': pipeline_physics,
            'pipeline_evolution': pipeline_evolution,
            'evolution_staged_reduction_gen50': evolution_staged_gen50,
            'evolution_staged_reduction_gen200': evolution_staged_gen200,
            'evolution_adaptive_expansion_gen50': evolution_adaptive_gen50,
        },
        'summary': {
            'aggressive_count': len(threshold_70),
            'moderate_count': len(threshold_60),
            'conservative_count': len(threshold_50),
            'tiered_core_count': len(tiered['core']),
            'pipeline_visual_count': len(pipeline_visual),
            'evolution_staged_gen50_count': len(evolution_staged_gen50),
            'evolution_staged_gen200_count': len(evolution_staged_gen200),
        }
    }
    
    # Save if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_path}")
    
    return report


# Example usage
if __name__ == "__main__":
    # Example: Blood vessel catheter
    example_instance = {
        "problem_id": "blood_vessel_catheter",
        "problem_class": "navigation",
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
    
    # Generate complete report
    report = generate_function_report(example_instance)
    
    print("\n" + "="*60)
    print("FUNCTION SELECTION REPORT")
    print("="*60)
    print(f"\nProblem: {report['problem_id']}")
    print(f"Class: {report['problem_class']}")
    
    print("\n--- Extracted Features ---")
    for key, value in report['extracted_features'].items():
        if value:  # Only show non-None, non-False, non-zero values
            print(f"  {key}: {value}")
    
    print("\n--- Top Function Weights ---")
    sorted_weights = sorted(report['function_weights'].items(), 
                          key=lambda x: x[1], 
                          reverse=True)
    for func, weight in sorted_weights[:10]:
        print(f"  {func}: {weight:.3f}")
    
    print("\n--- Recommended Function Sets ---")
    print(f"  Aggressive (T=0.07): {report['summary']['aggressive_count']} functions")
    print(f"    {report['recommendations']['aggressive']}")
    print(f"  Moderate (T=0.06): {report['summary']['moderate_count']} functions")
    print(f"    {report['recommendations']['moderate']}")
    print(f"  Conservative (T=0.05): {report['summary']['conservative_count']} functions")
    print(f"    {report['recommendations']['conservative']}")
    
    print("\n--- Pipeline-Based Selection (Validation Workflow) ---")
    print(f"  Visual inspection: {report['summary']['pipeline_visual_count']} functions")
    print(f"    {report['recommendations']['pipeline_visual_inspection']}")
    print(f"  Physics simulation: {len(report['recommendations']['pipeline_physics_simulation'])} functions")
    print(f"    {report['recommendations']['pipeline_physics_simulation']}")
    print(f"  Evolution: {len(report['recommendations']['pipeline_evolution'])} functions")
    print(f"    {report['recommendations']['pipeline_evolution']}")
    
    print("\n--- Evolution-Based Selection ---")
    print(f"  Strategy 4 - Staged Reduction Gen 50: {report['summary']['evolution_staged_gen50_count']} functions")
    print(f"    {report['recommendations']['evolution_staged_reduction_gen50']}")
    print(f"  Strategy 4 - Staged Reduction Gen 200: {report['summary']['evolution_staged_gen200_count']} functions")
    print(f"    {report['recommendations']['evolution_staged_reduction_gen200']}")
    print(f"  Strategy 5 - Adaptive Expansion Gen 50: {len(report['recommendations']['evolution_adaptive_expansion_gen50'])} functions")
    print(f"    {report['recommendations']['evolution_adaptive_expansion_gen50']}")
