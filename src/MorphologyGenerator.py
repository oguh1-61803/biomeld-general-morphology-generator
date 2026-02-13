# All the required imports and libraries.
from src.ActivationFunctionBank import ActivationFunctionBank
from src.DesignEngine import DesignEngine
from src.coordinate_systems import select_coordinate_system_from_ontology
from configupdater import ConfigUpdater
from src.function_selection import generate_function_report
import json
import neat


# This class orchestrates the generation of morphologies.
class MorphologyGenerator:

    NEAT_CONFIGURATION_PATH = "src/NEAT.cfg"
    # Since no evolutionary process is implied, the number of generations is 1.
    NUMBER_OF_GENERATIONS = 30

    def __init__(self, parameters_data):

        self.activation_function_bank = ActivationFunctionBank()
        self.cppn_design_engine = None

        self.__configure_file(parameters_data)
        self.__configure_cppn_design_engine(parameters_data)

    def GetTestCase(self, _data, _softbot_type): 
        for each_case in _data['test_cases']: 
            if (each_case['name'] == _softbot_type): 
                return each_case

        return _data['test_cases'][0]

    def select_active_functions(self, _file_name, _softbot_type, _algorithm_type): 
        print(f"[MorphologyGenerator] Selecting activation functions...")
        
        with open(_file_name, 'r') as f:
            data = json.load(f)

        example_instance = self.GetTestCase(data, _softbot_type)
        report = generate_function_report(example_instance)
        new_act_func = report['recommendations'][_algorithm_type]

        print(f"[MorphologyGenerator] Recommended functions: {new_act_func}")

        updater = ConfigUpdater()
        updater.read(self.NEAT_CONFIGURATION_PATH)
        
        activation_function_dictionary = ""
        for function in new_act_func:
            activation_function_dictionary += function + " "

        activation_function_dictionary = activation_function_dictionary[:-1]
        updater["DefaultGenome"]["activation_options"].value = activation_function_dictionary

        updater.update_file()
        
        print(f"[MorphologyGenerator] Updated NEAT.cfg with functions: {activation_function_dictionary}")

    def set_number_cppn(self, _num): 
        updater = ConfigUpdater()
        updater.read(self.NEAT_CONFIGURATION_PATH)
        updater["NEAT"]["pop_size"].value = _num
        updater.update_file()
        
        print(f"[MorphologyGenerator] Set population size to: {_num}")

        return 0

    # This method triggers the process to generate morphologies.
    def generate_morphologies(self):

        configuration = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation, self.NEAT_CONFIGURATION_PATH)

        # These lines add activation functions that are not built-in in the neat-python library. The activation
        # functions are contained in the activation function bank. To add one, it is compulsory to define it in the
        # ActivationFunctionBank.py file. Then, add it in the configuration. The following lines are examples of adding
        # custom activation functions to the activation function dictionary. The full list of built-in activation 
        # functions (excluding neg_abs, neg_square, sqrt_abs, neg_sqrt_abs, neg_sin) is in lines 23-25 of the Main.py file.
        configuration.genome_config.add_activation("neg_abs", self.activation_function_bank.negative_abs)
        configuration.genome_config.add_activation("neg_square", self.activation_function_bank.negative_square)
        configuration.genome_config.add_activation("sqrt_abs", self.activation_function_bank.square_abs)
        configuration.genome_config.add_activation("neg_sqrt_abs", self.activation_function_bank.negative_square_abs)
        configuration.genome_config.add_activation("neg_sin", self.activation_function_bank.negative_sin)

        population = neat.Population(configuration)
        reporter = neat.StdOutReporter(True)
        stats = neat.StatisticsReporter()
        population.add_reporter(reporter)
        population.add_reporter(stats)
        
        print(f"[MorphologyGenerator] Starting morphology generation...")
        print(f"[MorphologyGenerator] Population size: {configuration.pop_size}")
        print(f"[MorphologyGenerator] Generations: {self.NUMBER_OF_GENERATIONS}")
        
        population.run(self.cppn_design_engine.build_morphologies_using_cppns, self.NUMBER_OF_GENERATIONS)

        #population.stats.best_genome()
        
        print(f"[MorphologyGenerator] Morphology generation complete!")

    # This auxiliar method updates the .cfg file required by the neat-python library to generate CPPNs.
    def __configure_file(self, parameters_data):

        updater = ConfigUpdater()
        updater.read(self.NEAT_CONFIGURATION_PATH)

        updater["NEAT"]["pop_size"].value = parameters_data.get("number_of_cppns")

        if parameters_data.get("hidden_neurons") == 0:

            updater["DefaultGenome"]["initial_connection"].value = "full_direct"

            try:

                updater["DefaultGenome"]["num_hidden"].value = 0

            except KeyError:

                updater["DefaultGenome"]["num_inputs"].add_before.option("num_hidden", parameters_data.get("hidden_neurons"))

        else:

            updater["DefaultGenome"]["initial_connection"].value = "partial_direct 0.5"

            try:

                updater["DefaultGenome"]["num_hidden"].value = parameters_data.get("hidden_neurons")

            except KeyError:

                updater["DefaultGenome"]["num_inputs"].add_before.option("num_hidden", parameters_data.get("hidden_neurons"))

        activation_function_dictionary = ""

        for function in parameters_data.get("dictionary_of_activation_functions"):

            activation_function_dictionary += function + " "

        activation_function_dictionary = activation_function_dictionary[:-1]
        updater["DefaultGenome"]["activation_options"].value = activation_function_dictionary

        updater.update_file()

    # This auxiliar method initialises the CPPN design engine.
    def __configure_cppn_design_engine(self, parameters_data):
        
        # NEW: Auto-select coordinate system if not specified
        if 'coordinate_system' not in parameters_data:
            coord_system = select_coordinate_system_from_ontology(parameters_data)
            parameters_data['coordinate_system'] = coord_system
            print(f"[MorphologyGenerator] Auto-selected coordinate system: {coord_system}")
        else:
            print(f"[MorphologyGenerator] Using specified coordinate system: {parameters_data['coordinate_system']}")

        self.cppn_design_engine = DesignEngine(parameters_data)
