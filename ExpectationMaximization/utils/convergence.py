def convergence(
    val1: float | int, val2: float | int, epsilon: float
) -> bool:
    converged = False

    if abs(val1 - val2) <= epsilon:
        converged = True

    return converged
