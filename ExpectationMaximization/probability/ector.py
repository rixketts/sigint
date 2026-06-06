"""
Ector's Problem: An Introductory Example
"""
import math
import time
from scipy.optimize import minimize_scalar

from probability.ector_classes.dark import SquareObject, RoundObject, DarkObject
from probability.ector_classes.light import LightObject
from utils.validate import validate_x_vector, validate_p_param
from utils.results import ArgmaxG, EMRes

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


def expectation_maximization_algorithm_ector(
    y1: float | int,
    x3: float | int,
    p_0: float,
    max_iters: int = 1000,
    epsilon: float = 1e-6
) -> EMRes:
    """
    Using a warm-start p_0, find the values of p_k, x1_k, and x2_k
    (x3 is already known, so don't compute it here), where k = 1, 2, ..., max_iters.
    Repeat until max_iters is reached or the solver converges, whichever comes first

    :param x3: number of total light counts
    :type x3: nonneg float or int
    :param y1: number of total dark counts
    :type y1: nonneg float or int
    :param p_0: warmstart guess for finding p_k
    :type p_0: float
    :param max_iters: (optional) maximum iterations
    :type max_iters: int (default 1000)
    :param epsilon: (optional) convergence quantifier
    :type epsilon: float (default 1e-6)
    :return: p_k, x1_k, x2_k, and convergence status
    :rtype: ExpectationRes (float, float, float, bool respectively)
    """
    start_time = time.time()

    convergence = lambda last_val, second_to_last_val: abs(last_val - second_to_last_val) <= epsilon

    p_history: dict[str, float] = {}
    p_history["p_0"] = p_0

    x1_history: dict[str, float] = {}
    x2_history: dict[str, float] = {}

    converged = False

    iters = 1
    while iters <= max_iters:
        p_over_2 = p_history[f"p_{iters - 1}"] / 2.0

        # --- Expectation Step (E-step) ---
        # x_1^{k+1} = E[x_1|y_1,p^[k]]
        x1 = y1 * 0.25 / (0.5 + p_over_2)
        # x_2^{k+2} = E[x_1|y_1,p^{k}]
        x2 = y1 * (0.25 + p_over_2) / (0.5 + p_over_2)

        # --- Maximization Step (M-step) ---
        # p^[k+1] = (2x_2^{k+1} - x_3) / (x_2^{k+1} + x_3)
        p = (2.0 * x2 - x3) / (x2 + x3)

        x1_history[f"x1_{iters}"] = x1
        x2_history[f"x2_{iters}"] = x2
        p_history[f"p_{iters}"] = p

        if iters != 1:
            prev_x1 = x1_history[f"x1_{iters - 1}"]
            prev_x2 = x2_history[f"x2_{iters - 1}"]
            prev_p = p_history[f"p_{iters - 1}"]

            x1_converged = bool(convergence(x1, prev_x1))
            x2_converged = bool(convergence(x2, prev_x2))
            p_converged = bool(convergence(p, prev_p))

            if x1_converged and x2_converged and p_converged:
                converged = True

                return EMRes(
                    x1_k=x1, x2_k=x2, p_k=p, converged=converged, 
                    p_0=p_0, p_history=p_history, x1_history=x1_history,
                    x2_history=x2_history, solver_time=time.time() - start_time
                )
        
        iters += 1

        
    def _get_final_two_values(history: dict[str, float]) -> tuple[float, float]:
        """
        Get final two values from a histogram in the case that the while loop completes
        """
        values = list(history.values())
        
        return (values[-1], values[-2])
        

    last_x1, second_to_last_x1 = _get_final_two_values(x1_history)
    last_x2, second_to_last_x2 = _get_final_two_values(x2_history)
    last_p, second_to_last_p = _get_final_two_values(p_history)

    x1_converged = bool(convergence(last_x1, second_to_last_x1))
    x2_converged = bool(convergence(last_x2, second_to_last_x2))
    p_converged = bool(convergence(last_p, second_to_last_p))

    if x1_converged and x2_converged and p_converged:
        converged = True

    return EMRes(
        x1_k=last_x1, x2_k=last_x2, p_k=last_p, converged=converged,
        p_0=p_0, p_history=p_history, x1_history=x1_history,
        x2_history=x2_history, solver_time=time.time() - start_time
    )
