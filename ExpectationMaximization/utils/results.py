from dataclasses import dataclass


@dataclass
class ArgmaxG:
    """
    Results from taking argmax of g(Y1 = y1|p)
    """
    max_likelihood_estimate_p: float
    converged: bool
    solver_time: float


@dataclass
class EMRes:
    """
    Results from Expectation Maximization algorithm
    """
    x1_k: float
    x2_k: float
    p_k: float
    converged: bool
    p_0: float
    p_history: dict[str, float]
    x1_history: dict[str, float]
    x2_history: dict[str, float]
    solver_time: float


@dataclass
class EMResOneStep:
    """
    Results from a less computationally expensive rendition
    of Expectation Maximization Algorithm
    """
    p_k: float
    converged: bool
    p_0: float
    p_history: dict[str, float]
    solver_time: float
