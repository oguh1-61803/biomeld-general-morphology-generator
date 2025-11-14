from scipy.special import erf
import math

# This class represents the dictionary of functions that are not built-in in neat-python library.
class ActivationFunctionBank:

    def __init__(self):

        pass

    @staticmethod
    def negative_sin(z):

        # This if helps to avoid overflow.
        if abs(z) > 1.1377716103523e+307:

            return -math.sin(1.1377716103523e+307)

        return -math.sin(z)

    @staticmethod
    def negative_abs(z):

        return -abs(z)

    @staticmethod
    def negative_square(z):

        if abs(z) > 5.0e+20:

            return -z

        return -(z ** 2)

    @staticmethod
    def square_abs(z):

        return math.sqrt(abs(z))

    @staticmethod
    def negative_square_abs(z):

        return -math.sqrt(abs(z))

    # This activation function is not included in the dictionary. To include it, see lines 29-37 of the
    # MorphologyGenerator.py file.
    @staticmethod
    def cos(z):

        return math.cos(z)

    # This activation function is not included in the dictionary. To include it, see lines 29-37 of the
    # MorphologyGenerator.py file.
    @staticmethod
    def negative_cos(z):

        return -math.cos(z)

    # This activation function is not included in the dictionary. To include it, see lines 29-37 of the
    # MorphologyGenerator.py file.
    @staticmethod
    def gelu(z):

        cdf = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))

        return z * cdf
