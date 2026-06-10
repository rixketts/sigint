import math
import numpy as np
from typing import Iterable

from utils.validate import (
    validate_trinomial, validate_x_vector, validate_p_param
)


class Trinomial:
    def __init__(self) -> None:
        pass


    def get_joint_pmf(
        self,
        x_iterable: Iterable[int],
        p_iterable: Iterable[float]
    ) -> float:
        """
        Compute joint PMF for three classes
        
        :param x_iterable: realized values of counts for all classes
        :param p_iterable: class probabilities for all classes
        """
        validate_trinomial(x_iterable, p_iterable)

        x1, x2, x3 = tuple(x_iterable)
        p1, p2, p3 = tuple(p_iterable)

        num = math.factorial(x1 + x2 + x3)
        denom = math.factorial(x1) * math.factorial(x2) * math.factorial(x3)

        coef = (p1 ** x1) * (p2 ** x2) * (p3 ** x3)

        return (num / denom) * coef
    

    def get_joint_pmf_exponential(
        self,
        x_iterable: Iterable[int],
        p: float
    ) -> float:
        """
        Compute joint pmf using exponential form of function

        :param x_iterable: realized values of counts for all classes
        :param p: canonical parameter
        """
        validate_x_vector(x_iterable); validate_p_param(p)

        x = tuple(x_iterable)

        n = sum(x)

        factorial_coef = (
            math.factorial(n) / (
                math.factorial(x[0]) * math.factorial(x[1]) * math.factorial(x[2])
            )
        )

        log_vector = np.array([
            np.log((x[0] / 4.0) / (0.5 - p / 4.0)),
            np.log((0.25 + p / 4.0) / (0.5 - p / 4.0))
        ], dtype=np.float64)
        x_vector = np.array([[x[0]], [x[1]]], dtype=np.float64)

        exp_term = np.exp(log_vector @ x_vector)

        right_term = (0.5 - p / 4.0) ** n

        return float(factorial_coef * exp_term * right_term)
    

    def get_joint_pmf_combined_classes_x3(
        self,
        x_iterable: Iterable[int],
        p_iterable: Iterable[float]
    ) -> float:
        """
        Reduce three classes into two and compute joint PMF for X3 = x3

        :param x_iterable: realized values of counts for all classes
        :param p_iterable: class probabilities for all classes
        """
        validate_trinomial(x_iterable, p_iterable)

        x1, x2, x3 = tuple(x_iterable)
        p1, p2, p3 = tuple(p_iterable)

        y = x1 + x2

        factorial_coef = (
            math.factorial(y + x3) / (
                math.factorial(y) * math.factorial(x3)
            )
        )

        return factorial_coef * (p1 + p2) ** y * (p3 ** x3)
    

    def get_joint_pmf_combined_classes_x1(
        self,
        x_iterable: Iterable[int],
        p_iterable: Iterable[float]
    ) -> float:
        """
        Reduce three classes into two and compute joint PMF for X1 = x1

        :param x_iterable: realized values of counts for all classes
        :param p_iterable: class probabilities for all classes
        """
        validate_trinomial(x_iterable, p_iterable)

        x1, x2, x3 = tuple(x_iterable)
        p1, p2, p3 = tuple(p_iterable)

        y = x1 + x2

        factorial_coef = (
            math.factorial(y) / (
                math.factorial(x1) * math.factorial(y - x1)
            )
        )
        p_factor = (
            (p1 ** x1) * (p2 ** (y - x1)) / (p1 + p2) ** y
        )

        return factorial_coef * p_factor
    

    def conditional_expectation_x1(
        self,
        x_iterable: Iterable[int],
        p_iterable: Iterable[float]
    ) -> float:
        """
        Get average number of expected counts for x1

        :param x_iterable: realized values of counts for all classes
        :param p_iterable: class probabilities for all classes
        """
        validate_trinomial(x_iterable, p_iterable)

        x1, x2, x3 = tuple(x_iterable)
        p1, p2, p3 = tuple(p_iterable)

        y = x1 + x2

        return y * p1 / (p1 - p2)
    

    def conditional_expectation_x2(
        self,
        x_iterable: Iterable[int],
        p_iterable: Iterable[float]
    ) -> float:
        """
        Get average number of expected counts for x2

        :param x_iterable: realized values of counts for all classes
        :param p_iterable: class probabilities for all classes
        """
        validate_trinomial(x_iterable, p_iterable)

        x1, x2, x3 = tuple(x_iterable)
        p1, p2, p3 = tuple(p_iterable)

        y = x1 + x2

        return y * p2 / (p1 + p2)
