"""
Ector's Problem: An Introductory Example
"""
import math

from probability.ector_classes.dark import DarkObject
from probability.ector_classes.light import LightObject
from utils.validate import validate_x_vector, validate_p_param

DARK_OBJ = DarkObject()
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
    validate_x_vector(x); validate_p_param(p)

    n = sum(x)
    x1, x2, x3 = x

    multinomial_coef = (
        math.factorial(n) / (
            math.factorial(x1) * math.factorial(x2) * math.factorial(x3)
        )
    )

    # Probability of a round dark object, raised to the count
    # of round dark objects
    trinom_0 = DARK_OBJ.get_probability_round(p) ** x1
    # Probability of a square dark object, raised to the count
    # of square dark objects
    trinom_1 = DARK_OBJ.get_probability_square(p) ** x2
    # Probability of a light object, raised to the count of
    # light objects
    trinom_2 = LIGHT_OBJ.get_probability_light(p) ** x3

    return multinomial_coef * trinom_0 * trinom_1 * trinom_2
