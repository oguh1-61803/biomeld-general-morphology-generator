"""
Dynamic NEAT configuration for parameter testing and optimization.

Enables systematic testing of different NEAT parameters without manual .cfg editing.

Author: BioMeld Project
Date: December 2025
Version: 1.0
"""

from configupdater import ConfigUpdater
from typing import Dict, Any, List
import copy


class NEATConfigurator:
    """Manages NEAT configuration updates for testing/optimization."""
    
    DEFAULT_CONFIG_PATH = "src/NEAT.cfg"
    
    # Predefined parameter presets for quick testing
    PARAMETER_PRESETS = {
        'minimal': {
            'pop_size': 50,
            'num_hidden': 0,
            'initial_connection': 'full_direct',
            'weight_mutate_rate': 0.6,
            'conn_add_prob': 0.2,
            'conn_delete_prob': 0.1,
            'node_add_prob': 0.1,
            'node_delete_prob': 0.05,
        },
        'standard': {
            'pop_size': 100,
            'num_hidden': 2,
            'initial_connection': 'partial_direct 0.5',
            'weight_mutate_rate': 0.8,
            'conn_add_prob': 0.3,
            'conn_delete_prob': 0.2,
            'node_add_prob': 0.3,
            'node_delete_prob': 0.2,
        },
        'complex': {
            'pop_size': 150,
            'num_hidden': 4,
            'initial_connection': 'partial_direct 0.3',
            'weight_mutate_rate': 0.9,
            'conn_add_prob': 0.4,
            'conn_delete_prob': 0.3,
            'node_add_prob': 0.4,
            'node_delete_prob': 0.3,
        },
        'conservative': {
            'pop_size': 100,
            'num_hidden': 2,
            'initial_connection': 'partial_direct 0.5',
            'weight_mutate_rate': 0.5,
            'conn_add_prob': 0.2,
            'conn_delete_prob': 0.1,
            'node_add_prob': 0.2,
            'node_delete_prob': 0.1,
        },
        'aggressive': {
            'pop_size': 100,
            'num_hidden': 3,
            'initial_connection': 'partial_direct 0.4',
            'weight_mutate_rate': 1.0,
            'conn_add_prob': 0.5,
            'conn_delete_prob': 0.4,
            'node_add_prob': 0.5,
            'node_delete_prob': 0.4,
        }
    }
    
    def __init__(self, config_path: str = None):
        """
        Initialize NEAT configurator.
        
        Args:
            config_path: Path to NEAT.cfg file (default: src/NEAT.cfg)
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.updater = ConfigUpdater()
        self.updater.read(self.config_path)
    
    def apply_preset(self, preset_name: str):
        """
        Apply a predefined parameter preset.
        
        Args:
            preset_name: Name of preset ('minimal', 'standard', 'complex', 
                        'conservative', 'aggressive')
        
        Raises:
            ValueError: If preset_name is not recognized
        """
        if preset_name not in self.PARAMETER_PRESETS:
            available = ', '.join(self.PARAMETER_PRESETS.keys())
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {available}"
            )
        
        params = self.PARAMETER_PRESETS[preset_name]
        self.update_parameters(params)
        print(f"[NEATConfigurator] Applied preset: {preset_name}")
    
    def update_parameters(self, params: Dict[str, Any]):
        """
        Update specific NEAT parameters.
        
        Args:
            params: Dictionary of parameter names and values
        
        Example:
            >>> configurator.update_parameters({
            ...     'pop_size': 150,
            ...     'num_hidden': 3,
            ...     'weight_mutate_rate': 0.9
            ... })
        """
        for key, value in params.items():
            # NEAT section parameters
            if key in ['pop_size', 'fitness_threshold', 'reset_on_extinction']:
                self.updater["NEAT"][key].value = value
            
            # DefaultGenome section parameters
            elif key in ['num_hidden', 'weight_mutate_rate', 'conn_add_prob', 
                         'conn_delete_prob', 'node_add_prob', 'node_delete_prob',
                         'initial_connection', 'bias_mutate_rate', 'response_mutate_rate',
                         'activation_mutate_rate']:
                
                if key == 'num_hidden':
                    # num_hidden may not exist in config, need special handling
                    try:
                        self.updater["DefaultGenome"][key].value = value
                    except KeyError:
                        # Add if doesn't exist
                        self.updater["DefaultGenome"]["num_inputs"].add_before.option(
                            "num_hidden", value
                        )
                else:
                    self.updater["DefaultGenome"][key].value = value
            
            else:
                print(f"[NEATConfigurator] Warning: Unknown parameter '{key}' ignored")
    
    def update_activation_functions(self, function_list: List[str]):
        """
        Update activation function dictionary.
        
        Args:
            function_list: List of activation function names
        
        Example:
            >>> configurator.update_activation_functions(['sin', 'gauss', 'sigmoid'])
        """
        function_str = " ".join(function_list)
        self.updater["DefaultGenome"]["activation_options"].value = function_str
        print(f"[NEATConfigurator] Updated activation functions: {function_str}")
    
    def set_population_size(self, size: int):
        """
        Quick setter for population size.
        
        Args:
            size: Population size (number of genomes)
        """
        self.updater["NEAT"]["pop_size"].value = size
        print(f"[NEATConfigurator] Set population size to: {size}")
    
    def set_hidden_neurons(self, count: int):
        """
        Quick setter for number of hidden neurons.
        
        Args:
            count: Number of hidden neurons
        """
        try:
            self.updater["DefaultGenome"]["num_hidden"].value = count
        except KeyError:
            self.updater["DefaultGenome"]["num_inputs"].add_before.option(
                "num_hidden", count
            )
        
        # Adjust initial connection based on hidden neuron count
        if count == 0:
            self.updater["DefaultGenome"]["initial_connection"].value = "full_direct"
        else:
            self.updater["DefaultGenome"]["initial_connection"].value = "partial_direct 0.5"
        
        print(f"[NEATConfigurator] Set hidden neurons to: {count}")
    
    def save(self):
        """Write updated configuration to file."""
        self.updater.update_file()
        print(f"[NEATConfigurator] Configuration saved to: {self.config_path}")
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current configuration as dictionary.
        
        Returns:
            Dictionary with current parameter values
        """
        try:
            num_hidden = int(self.updater["DefaultGenome"].get("num_hidden", 0).value)
        except (KeyError, AttributeError):
            num_hidden = 0
        
        return {
            'pop_size': int(self.updater["NEAT"]["pop_size"].value),
            'num_hidden': num_hidden,
            'initial_connection': self.updater["DefaultGenome"]["initial_connection"].value,
            'weight_mutate_rate': float(self.updater["DefaultGenome"]["weight_mutate_rate"].value),
            'conn_add_prob': float(self.updater["DefaultGenome"]["conn_add_prob"].value),
            'conn_delete_prob': float(self.updater["DefaultGenome"]["conn_delete_prob"].value),
            'activation_functions': self.updater["DefaultGenome"]["activation_options"].value.split()
        }
    
    def print_current_config(self):
        """Print current configuration in readable format."""
        config = self.get_current_config()
        print("\n" + "="*60)
        print("Current NEAT Configuration:")
        print("="*60)
        for key, value in config.items():
            if key == 'activation_functions':
                print(f"{key:25s}: {', '.join(value)}")
            else:
                print(f"{key:25s}: {value}")
        print("="*60 + "\n")


class ParameterTester:
    """Systematic testing of NEAT parameter combinations."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize parameter tester.
        
        Args:
            config_path: Path to NEAT.cfg file
        """
        self.config_path = config_path
        self.results = []
    
    def test_hidden_neuron_counts(self, counts: List[int], base_config: Dict[str, Any] = None):
        """
        Generate configurations for testing different hidden neuron counts.
        
        Args:
            counts: List of hidden neuron counts to test
            base_config: Base configuration (optional)
        
        Yields:
            Configuration dictionary for each test
        
        Example:
            >>> tester = ParameterTester()
            >>> for config in tester.test_hidden_neuron_counts([0, 2, 4, 6]):
            ...     print(f"Testing {config['num_hidden']} hidden neurons")
        """
        if base_config is None:
            base_config = {'pop_size': 100}
        
        for count in counts:
            config = copy.deepcopy(base_config)
            config['num_hidden'] = count
            
            configurator = NEATConfigurator(self.config_path)
            configurator.update_parameters(config)
            configurator.save()
            
            print(f"[ParameterTester] Testing hidden neurons: {count}")
            yield config
    
    def test_population_sizes(self, sizes: List[int], base_config: Dict[str, Any] = None):
        """
        Generate configurations for testing different population sizes.
        
        Args:
            sizes: List of population sizes to test
            base_config: Base configuration (optional)
        
        Yields:
            Configuration dictionary for each test
        """
        if base_config is None:
            base_config = {'num_hidden': 2}
        
        for size in sizes:
            config = copy.deepcopy(base_config)
            config['pop_size'] = size
            
            configurator = NEATConfigurator(self.config_path)
            configurator.update_parameters(config)
            configurator.save()
            
            print(f"[ParameterTester] Testing population size: {size}")
            yield config
    
    def test_mutation_rates(self, rates: List[float], base_config: Dict[str, Any] = None):
        """
        Generate configurations for testing different mutation rates.
        
        Args:
            rates: List of weight mutation rates to test
            base_config: Base configuration (optional)
        
        Yields:
            Configuration dictionary for each test
        """
        if base_config is None:
            base_config = {'pop_size': 100, 'num_hidden': 2}
        
        for rate in rates:
            config = copy.deepcopy(base_config)
            config['weight_mutate_rate'] = rate
            
            configurator = NEATConfigurator(self.config_path)
            configurator.update_parameters(config)
            configurator.save()
            
            print(f"[ParameterTester] Testing mutation rate: {rate}")
            yield config


# Testing and examples
if __name__ == "__main__":
    print("NEAT Configurator - Testing Module")
    print("="*60)
    
    # Create configurator
    configurator = NEATConfigurator()
    
    # Show current config
    print("\n1. Current Configuration:")
    configurator.print_current_config()
    
    # Test preset application
    print("\n2. Testing Presets:")
    for preset_name in ['minimal', 'standard', 'complex']:
        print(f"\nApplying preset: {preset_name}")
        configurator.apply_preset(preset_name)
        current = configurator.get_current_config()
        print(f"  Population: {current['pop_size']}")
        print(f"  Hidden neurons: {current['num_hidden']}")
        print(f"  Mutation rate: {current['weight_mutate_rate']}")
    
    # Test manual updates
    print("\n3. Testing Manual Updates:")
    configurator.update_parameters({
        'pop_size': 75,
        'num_hidden': 3,
        'weight_mutate_rate': 0.7
    })
    configurator.print_current_config()
    
    print("\n✅ All tests completed!")
