# All the requeried imports and libraries.
from src.ActivationFunctionBank import ActivationFunctionBank
from src.DesignEngine import DesignEngine
from configupdater import ConfigUpdater
import neat


# This class orchestrates the generation of morphologies.
class MorphologyGenerator:

    NEAT_CONFIGURATION_PATH = "NEAT.cfg"
    # Since no evolutionary process is implied, the number of generations is 1.
    NUMBER_OF_GENERATIONS = 1

    def __init__(self, parameters_data):

        self.activation_function_bank = ActivationFunctionBank()
        self.cppn_design_engine = None

        self.__configure_file(parameters_data)
        self.__configure_cppn_design_engine(parameters_data)

    # This method triggers the process to generate morphologies.
    def generate_morphologies(self):

        configuration = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation, self.NEAT_CONFIGURATION_PATH)

        # These lines add activation functions that are not built-in in the neat-python library. The activation
        # functions are contained in the activation function bank. To add one, it is compulsory to define it in the
        # ActivationFunctionBank.py file. Then, add it in the configuration. The following lines are examples of adding
        # custom activation functions to the activation function dictionary.
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
        population.run(self.cppn_design_engine.build_morphologies_using_cppns, self.NUMBER_OF_GENERATIONS)

    # This auxiliar method updates the .cfg file required by the neat-python library to generate CPPNs.
    def __configure_file(self, parameters_data):

        updater = ConfigUpdater()
        updater.read(self.NEAT_CONFIGURATION_PATH)

        if parameters_data.get("hidden_neurons") == 0:

            updater["DefaultGenome"]["initial_connection"].value = "full_direct"

            try:

                updater["DefaultGenome"]["num_hidden"].value = 0

            except KeyError:

                updater["DefaultGenome"]["num_inputs"].add_before.option("num_hidden", parameters_data.get("hidden_neurons"))

        else:

            updater["DefaultGenome"]["initial_connection"].value =  "partial_direct 0.5"

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

        self.cppn_design_engine = DesignEngine(parameters_data)