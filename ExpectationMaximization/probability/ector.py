"""
Ector's Problem: An Introductory Example
"""
import math

from probability.ector_classes.dark import SquareObject, RoundObject, DarkObject
from probability.ector_classes.light import LightObject
from utils.validate import validate_x_vector, validate_p_param

DARK_OBJ = DarkObject()
DARK_SQUARE_OBJ = SquareObject()
DARK_ROUND_OBJ = RoundObject()

LIGHT_OBJ = LightObject()


def trinomial_probability_det_objs(
    x: tuple[int, int, int],
    p: float
) -> float:
    """
    Simple trinomial probability computation to calculate
    the probability of detecting x1 round dark objects,
    x2 square dark objects, and x3 light objects in
    an arbitrary image

    :param x: (x1, x2, x3)
    :type x: tuple of three nonneg ints
    :param p: input for parameter of the distribution
    :type p: float between -1 and 2
    """
    n = sum(x)
    x1, x2, x3 = x

    multinomial_coef = (
        math.factorial(n) / (
            math.factorial(x1) * math.factorial(x2) * math.factorial(x3)
        )
    )

    # Probability of a round dark object, raised to the count
    # of round dark objects
    trinom_1 = DARK_ROUND_OBJ.get_probability(p) ** x1
    # Probability of a square dark object, raised to the count
    # of square dark objects
    trinom_2 = DARK_SQUARE_OBJ.get_probability(p) ** x2
    # Probability of a light object, raised to the count of
    # light objects
    trinom_3 = LIGHT_OBJ.get_probability(p) ** x3

    return multinomial_coef * trinom_1 * trinom_2 * trinom_3


def feature_extractor(
    x: tuple[int, int, int],
    p: float
) -> float:
    """
    Shape-agnostic feature extractor that calculates
    the probability of detecting y1 dark objects and
    y2 light objects via a binomial probability computation

    :param x: (x1, x2, x3) NOTE: still pass in the tuple
              of round dark objects, square dark objects, and light objects
    :type x: tuple of three nonneg ints
    :param p: input for parameter of the distribution
    :type p: float between -1 and 2
    """
    y1 = x[0] + x[1]
    y2 = x[2]

    n = y1 + y2

    # (n \\ y1)
    binomial_coef = (
        math.factorial(n) / (
            math.factorial(y1) * math.factorial(n - y1)
        )
    )

    # Probability of a dark object, raised to the count of
    # dark objects
    binom_1 = DARK_OBJ.get_probability(p) ** y1
    # Probability of a light object, raised to the count of
    # light objects
    binom_2 = LIGHT_OBJ.get_probability(p) ** (n - y1)  # y2 = n - y1

    return binomial_coef * binom_1 * binom_2
