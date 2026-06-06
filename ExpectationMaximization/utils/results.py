from dataclasses import dataclass


@dataclass
class ArgmaxG:
    max_likelihood_estimate_p: float
    converged: bool
    solver_time: float
