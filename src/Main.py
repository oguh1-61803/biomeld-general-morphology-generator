# All the requeried imports.
from src.MorphologyGenerator import MorphologyGenerator


# Everything starts here.
if __name__ == '__main__':

    #JSON containing the necessary parameters to generate CPPNs.
    parameters_data = {
        # The layout dimensions in terms voxels.
        "dimensions_3d_layout":{
            "x":15,
            "y":8,
            "z":10
        },
        # Number of CPPNs (and morphologies) that will be generated.
        "number_of_cppns": 3,
        # Number of hidden neurons of CPPNs. If 0, the input neurons will be connected to the output neurons. If >0, the
        # connections are randomised with 0.5 of probability to be enabled.
        "hidden_neurons": 2,
        # All the activation functions envolved in the creation of CPPNs.
        "dictionary_of_activation_functions": [
            "sin", "neg_sin", "abs", "neg_abs", "square", "neg_square", "sqrt_abs", "neg_sqrt_abs", "sigmoid",
            "clamped", "cube", "exp", "gauss", "hat", "identity", "inv", "log", "relu", "selu", "elu", "lelu", "tanh",
            "softplus"
        ]
    }

    mg = MorphologyGenerator(parameters_data)
    mg.generate_morphologies()
