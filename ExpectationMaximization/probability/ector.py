"""
Ector's Problem: An Introductory Example
"""
import math
import time
from scipy.optimize import minimize_scalar

from probability.ector_classes.dark import SquareObject, RoundObject, DarkObject
from probability.ector_classes.light import LightObject
from utils.validate import validate_x_vector, validate_p_param
from utils.results import ArgmaxG

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
    y1: int, y2: int, p: float
) -> float:
    """
    Shape-agnostic feature extractor that calculates
    the probability of detecting y1 dark objects and
    y2 light objects via a binomial probability computation

    :param y1: number of dark objects
    :type y1: nonneg int
    :param y2: number of light objects
    :type y2: nonneg int
    :param p: input for parameter of the distribution
    :type p: float between -1 and 2
    """
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


def p_ml_argmax_prob_fn(y1: int, y2: int) -> ArgmaxG:
    """
    Use a scalar-minimizing optimization function to
    find the minimum value of the negative of the feature
    extractor (thus, finding the maximum value of the original extractor)

    :param y1: number of dark objects
    :type y1: nonneg int
    :param y2: number of light objects
    :type y2: nonneg int
    :return: optimal p-value and solver convergence status
    """
    start_time = time.time()

    solver_result = minimize_scalar(
        fun=lambda p: -1.0 * feature_extractor(y1, y2, p),
        bounds=(-1.0, 2.0),
        method="bounded"
    )

    end_time = time.time() - start_time

    return ArgmaxG(max_likelihood_estimate_p=float(solver_result.x),
                   converged=bool(solver_result.success),
                   solver_time=end_time)
