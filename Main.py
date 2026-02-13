# All the required imports.
from src.MorphologyGenerator import MorphologyGenerator
from src.neat_configurator import NEATConfigurator, ParameterTester
from src.coordinate_systems import test_coordinate_transformation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import json


# ============================================================================
# SYSTEMATIC TESTING SUITE
# ============================================================================

def test_coordinate_systems():
    """
    Test 1: Validate coordinate system transformations.
    Generate morphologies using different coordinate systems and compare results.
    """
    print("\n" + "="*70)
    print("TEST 1: COORDINATE SYSTEM VALIDATION")
    print("="*70)
    
    # Base parameters for testing
    base_params = {
        "dimensions_3d_layout": {"x": 15, "y": 8, "z": 10},
        "number_of_cppns": 3,
        "hidden_neurons": 2,
        "dictionary_of_activation_functions": [
            "sin", "neg_sin", "abs", "neg_abs", "gauss", "sigmoid", "tanh"
        ]
    }
    
    coordinate_systems = ['cartesian', 'cylindrical', 'spherical']
    
    for coord_system in coordinate_systems:
        print(f"\n--- Testing {coord_system.upper()} coordinate system ---")
        
        # Create parameters with specific coordinate system
        test_params = base_params.copy()
        test_params['coordinate_system'] = coord_system
        
        # Generate morphologies
        print(f"[Test] Generating {test_params['number_of_cppns']} morphologies...")
        start_time = time.time()
        
        mg = MorphologyGenerator(test_params)
        mg.generate_morphologies()
        
        elapsed_time = time.time() - start_time
        print(f"[Test] {coord_system.capitalize()} system completed in {elapsed_time:.2f} seconds")
        print(f"[Test] Check output in: src/morphologies/morphology_0/, morphology_1/, morphology_2/")
        print()
    
    print("✅ Coordinate system test complete!")
    print("   Review generated .vxa files in src/morphologies/ directory")
    print()


def test_archetype_coordinate_selection():
    """
    Test 2: Validate automatic coordinate system selection based on archetypes.
    """
    print("\n" + "="*70)
    print("TEST 2: ARCHETYPE-BASED COORDINATE SYSTEM SELECTION")
    print("="*70)
    
    # Test cases for different archetypes
    test_cases = [
        {
            'name': 'Archetype 1 (Elongated Cylinder)',
            'params': {
                'archetype': 1,
                'symmetry_type': 'radial',
                'base_morphology': 'elongated_cylinder',
                'dimensions_3d_layout': {'x': 15, 'y': 8, 'z': 60},
                'number_of_cppns': 2,
                'hidden_neurons': 2,
                'dictionary_of_activation_functions': ['sin', 'neg_sin', 'gauss', 'neg_abs', 'sigmoid']
            },
            'expected_system': 'cylindrical'
        },
        {
            'name': 'Archetype 3 (Sphere)',
            'params': {
                'archetype': 3,
                'symmetry_type': 'spherical',
                'base_morphology': 'sphere',
                'dimensions_3d_layout': {'x': 20, 'y': 20, 'z': 20},
                'number_of_cppns': 2,
                'hidden_neurons': 2,
                'dictionary_of_activation_functions': ['sin', 'neg_sin', 'gauss', 'neg_abs']
            },
            'expected_system': 'spherical'
        },
        {
            'name': 'Archetype 4 (Branched)',
            'params': {
                'archetype': 4,
                'symmetry_type': 'bilateral',
                'base_morphology': 'branched',
                'dimensions_3d_layout': {'x': 15, 'y': 15, 'z': 10},
                'number_of_cppns': 2,
                'hidden_neurons': 2,
                'dictionary_of_activation_functions': ['gauss', 'abs', 'relu', 'sigmoid']
            },
            'expected_system': 'cartesian'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing {test_case['name']} ---")
        print(f"[Test] Expected coordinate system: {test_case['expected_system']}")
        
        mg = MorphologyGenerator(test_case['params'])
        
        # Verify coordinate system was selected correctly
        actual_system = mg.cppn_design_engine.coordinate_transformer.system
        print(f"[Test] Actual coordinate system: {actual_system}")
        
        if actual_system == test_case['expected_system']:
            print(f"[Test] ✅ PASS - Correct coordinate system selected")
        else:
            print(f"[Test] ❌ FAIL - Expected {test_case['expected_system']}, got {actual_system}")
        
        # Generate morphologies
        print(f"[Test] Generating morphologies...")
        mg.generate_morphologies()
        print()
    
    print("✅ Archetype coordinate selection test complete!")
    print()


def test_neat_configurations():
    """
    Test 3: Test different NEAT configuration presets.
    """
    print("\n" + "="*70)
    print("TEST 3: NEAT CONFIGURATION PRESETS")
    print("="*70)
    
    base_params = {
        "dimensions_3d_layout": {"x": 15, "y": 8, "z": 10},
        "number_of_cppns": 5,
        "hidden_neurons": 2,  # Will be overridden by preset
        "coordinate_system": "cylindrical",
        "dictionary_of_activation_functions": [
            "sin", "gauss", "neg_abs", "sigmoid", "identity"
        ]
    }
    
    presets = ['minimal', 'standard', 'complex']
    
    configurator = NEATConfigurator()
    
    for preset_name in presets:
        print(f"\n--- Testing {preset_name.upper()} preset ---")
        
        # Apply preset
        configurator.apply_preset(preset_name)
        configurator.save()
        
        # Show configuration
        config = configurator.get_current_config()
        print(f"[Test] Population size: {config['pop_size']}")
        print(f"[Test] Hidden neurons: {config['num_hidden']}")
        print(f"[Test] Mutation rate: {config['weight_mutate_rate']}")
        
        # Generate morphologies with this configuration
        print(f"[Test] Generating morphologies...")
        start_time = time.time()
        
        mg = MorphologyGenerator(base_params)
        mg.generate_morphologies()
        
        elapsed_time = time.time() - start_time
        print(f"[Test] Completed in {elapsed_time:.2f} seconds")
        print()
    
    print("✅ NEAT configuration test complete!")
    print()


def test_hidden_neuron_variations():
    """
    Test 4: Systematically test different hidden neuron counts.
    """
    print("\n" + "="*70)
    print("TEST 4: HIDDEN NEURON COUNT VARIATIONS")
    print("="*70)
    
    base_params = {
        "dimensions_3d_layout": {"x": 15, "y": 8, "z": 10},
        "number_of_cppns": 3,
        "coordinate_system": "cylindrical",
        "dictionary_of_activation_functions": [
            "sin", "neg_sin", "gauss", "neg_abs", "sigmoid"
        ]
    }
    
    hidden_neuron_counts = [0, 2, 4]
    
    configurator = NEATConfigurator()
    
    for count in hidden_neuron_counts:
        print(f"\n--- Testing {count} hidden neurons ---")
        
        # Update configuration
        configurator.set_hidden_neurons(count)
        configurator.save()
        
        # Update parameters
        test_params = base_params.copy()
        test_params['hidden_neurons'] = count
        
        # Generate morphologies
        print(f"[Test] Generating morphologies...")
        start_time = time.time()
        
        mg = MorphologyGenerator(test_params)
        mg.generate_morphologies()
        
        elapsed_time = time.time() - start_time
        print(f"[Test] Completed in {elapsed_time:.2f} seconds")
        print()
    
    print("✅ Hidden neuron variation test complete!")
    print()


def test_combined_system():
    """
    Test 5: Full integration test with coordinate system auto-selection 
    and NEAT configuration.
    """
    print("\n" + "="*70)
    print("TEST 5: FULL INTEGRATION TEST")
    print("="*70)
    
    # Simulate blood vessel catheter (Archetype 1)
    print("\n--- Test Case: Blood Vessel Catheter ---")
    
    catheter_params = {
        "dimensions_3d_layout": {"x": 15, "y": 8, "z": 60},
        "archetype": 1,
        "symmetry_type": "radial",
        "base_morphology": "elongated_cylinder",
        "aspect_ratio": 20,
        "hollowness_type": "hollow",
        "number_of_cppns": 5,
        "hidden_neurons": 2,
        "dictionary_of_activation_functions": [
            "sin", "neg_sin", "gauss", "sigmoid", "neg_abs", "tanh", "identity"
        ]
    }
    
    # Configure NEAT
    configurator = NEATConfigurator()
    configurator.apply_preset('standard')
    configurator.save()
    
    print(f"[Test] Generating catheter morphologies...")
    start_time = time.time()
    
    mg = MorphologyGenerator(catheter_params)
    
    # Verify auto-selection
    selected_system = mg.cppn_design_engine.coordinate_transformer.system
    print(f"[Test] Auto-selected coordinate system: {selected_system}")
    print(f"[Test] Expected: cylindrical")
    
    if selected_system == 'cylindrical':
        print(f"[Test] ✅ Correct coordinate system!")
    else:
        print(f"[Test] ⚠️ Unexpected coordinate system!")
    
    # Generate
    mg.generate_morphologies()
    
    elapsed_time = time.time() - start_time
    print(f"[Test] Integration test completed in {elapsed_time:.2f} seconds")
    print()
    
    print("✅ Full integration test complete!")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    """ 
    print("\n" + "="*70)
    print("BIOMELD MORPHOLOGY GENERATION - SYSTEMATIC TESTING SUITE")
    print("="*70)
    print("\nThis script will systematically test:")
    print("  1. Coordinate system transformations (Cartesian, Cylindrical, Spherical)")
    print("  2. Archetype-based coordinate system auto-selection")
    print("  3. NEAT configuration presets (minimal, standard, complex)")
    print("  4. Hidden neuron count variations (0, 2, 4)")
    print("  5. Full integration test (Blood Vessel Catheter example)")
    print("\nPress Enter to start testing, or Ctrl+C to cancel...")
    input()
    
    # Run all tests
    try:
        # Test 1: Basic coordinate system validation
        test_coordinate_systems()
        
        # Test 2: Archetype-based selection
        test_archetype_coordinate_selection()
        
        # Test 3: NEAT configuration presets
        test_neat_configurations()
        
        # Test 4: Hidden neuron variations
        test_hidden_neuron_variations()
        
        # Test 5: Full integration
        test_combined_system()
        
        # Summary
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nResults saved in: src/morphologies/")
        print("Review the generated .vxa files to visually inspect morphologies")
        print("\nNext steps:")
        print("  1. Open generated .vxa files in VoxCAD")
        print("  2. Visually compare morphologies from different coordinate systems")
        print("  3. Analyze differences from NEAT configuration changes")
        print("  4. Proceed to implement fitness functions and evolution")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    """


    
    parameters_data = {
        "dimensions_3d_layout": {"x": 15, "y": 8, "z": 10},
        "number_of_cppns": 3,
        "hidden_neurons": 2,                
	'symmetry_type': 'radial',
        'base_morphology': 'elongated_cylinder',
        'fitness_shape': 'cylinder',
        "coordinate_system": "cylindrical",  # Try: 'cartesian', 'cylindrical', 'spherical'
        "dictionary_of_activation_functions": [
            "identity","sine", "negative_sine", "sigmoid"
        ]
    }
    

    """
    parameters_data = {
        "dimensions_3d_layout": {"x": 15, "y": 8, "z": 10},
        "number_of_cppns": 3,
        "hidden_neurons": 2,                
	'symmetry_type': 'sperical',
        'base_morphology': 'elongated_cylinder',
        'fitness_shape': 'sphere',
        "coordinate_system": "spherical",  # Try: 'cartesian', 'cylindrical', 'spherical'
        "dictionary_of_activation_functions": [
            "gauss","sin", "neg_sin", "sigmoid" 
        ]
    }
    """

    """
    parameters_data = {
        "dimensions_3d_layout": {"x": 15, "y": 8, "z": 10},
        "number_of_cppns": 3,
        "hidden_neurons": 2,                
	'symmetry_type': 'bilateral',
        'base_morphology': 'elongated_cylinder',
        'fitness_shape': 'cube',
        "coordinate_system": "cartesian",  # Try: 'cartesian', 'cylindrical', 'spherical'
        "dictionary_of_activation_functions": [
            "square","abs", "relu", "neg_abs" 
        ]
    }
    """

    ff = open('log.html', 'w')
    ff.write('{')
    ff.close()
    ff = open('metrics.html', 'w')
    ff.write('{')
    ff.close()

    mg = MorphologyGenerator(parameters_data) #,"cylinder",'aggressive')
    mg.select_active_functions("src/test_case_ontology_specifications.json","cylinder",'aggressive')
    mg.set_number_cppn(100)
    mg.generate_morphologies()


    ff = open('log.html', 'a')

    ff.write('"null":')
    ff.write('["000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"]     }')
    ff.close()






    with open('log.html', 'r') as f:
            data = json.load(f)

















    num_plot = 0
    num_plot2 = 0
    #if (len(data) > 100) : num_plot2 = len(data) - 100
    fig = plt.figure(figsize=(44, 44))
    #for each in data[-100:] : 
    for each in data : 
        num_plot2 = num_plot2 + 1
        if (num_plot2 > len(data) - 100) : 
          num_plot = num_plot + 1

          if (num_plot > 100) : break

        # build up the numpy logo
          n_voxels = np.zeros((15, 8, 60), dtype=bool)


          xs = []
          ys = []
          zs = []
          xs2 = []
          ys2 = []
          zs2 = []
          z_Num = 0
          for eachlayer in data[each] : 
            xy_Num = 0
            for xy in eachlayer : 
                y_Num = int(xy_Num/15)
                x_Num = int(xy_Num) - 15*y_Num
                if (xy == "1") :         
                    n_voxels[x_Num, y_Num, z_Num] = 1
                    xs.append(x_Num)
                    ys.append(y_Num)
                    zs.append(z_Num)
                if (xy == "3") :         
                    n_voxels[x_Num, y_Num, z_Num] = 3
                    xs2.append(x_Num)
                    ys2.append(y_Num)
                    zs2.append(z_Num)
                xy_Num = xy_Num + 1
            z_Num = z_Num + 1





          ax = fig.add_subplot(10,10,num_plot, projection='3d')
          ax.text(-10, 10,10, each, size=10, color='purple')
          #ax.voxels(x, y, z, filled_2, facecolors='#00000000', edgecolor='#00000000')
          ax.scatter(xs, ys, zs)
          ax.scatter(xs2, ys2, zs2)
          ax.scatter([0,15], [0,8], [0, 10])
          #ax.set_aspect('auto')




    plt.savefig('snapshot.png')
    plt.close()











# ============================================================================
# ALTERNATIVE: QUICK SINGLE TEST
# ============================================================================

def quick_test():
    """
    Quick test for rapid iteration during development.
    Generates just a few morphologies for quick visual inspection.
    """
    print("\n" + "="*70)
    print("QUICK TEST MODE")
    print("="*70)
    
    # Minimal test parameters
    parameters_data = {
        "dimensions_3d_layout": {"x": 15, "y": 8, "z": 10},
        "number_of_cppns": 3,
        "hidden_neurons": 2,
        "coordinate_system": "cylindrical",  # Try: 'cartesian', 'cylindrical', 'spherical'
        "dictionary_of_activation_functions": [
            "sin", "neg_sin", "gauss", "neg_abs", "sigmoid"
        ]
    }
    
    print(f"\n[QuickTest] Coordinate system: {parameters_data['coordinate_system']}")
    print(f"[QuickTest] Generating {parameters_data['number_of_cppns']} morphologies...")
    
    mg = MorphologyGenerator(parameters_data)
    mg.generate_morphologies()
    
    print("\n✅ Quick test complete!")
    print("Check: src/morphologies/morphology_0/morphology.vxa")


# Uncomment this to run quick test instead:
# if __name__ == '__main__':
#     quick_test()
