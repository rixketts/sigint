from probability.ector import (
    trinomial_probability_det_objs, feature_extractor, p_ml_argmax_prob_fn
)
from utils.validate import validate_x_vector, validate_p_param
from utils.results import ArgmaxG

import sys


def print_usage(
    usage: str = "python ector_intro_script.py x1 x2 x3 p"
) -> None:
    if len(sys.argv) != 5:
        print("Usage:", usage)

        sys.exit(0)


def trinomial_pipeline(
    x: tuple[int, int, int], p: float
) -> float:
    """
    Compute initial probability using simple trinomial function
    """
    prob_trinom = trinomial_probability_det_objs(x, p)
    
    print(
        f"\nTrinom. probability of finding {x[0]} dark round objs, "
        f"{x[1]} dark square objs, {x[2]} light objs: {prob_trinom:.6f}"
    )

    return prob_trinom


def feature_extractor_pipeline(
    y1: int, y2: int, p: float, prob_trinom: float
) -> tuple[float, float]:
    """
    Compute initial probability using binomial from feature
    extractor and return binomial probabilty and percent difference
    between binom. and trinom. prob. wrt trinom
    """
    prob_binom = feature_extractor(y1, y2, p)
    
    print(
        f"\nBinom. probability of finding {y1} dark round objs "
        f"and {y2} light objs: {prob_binom:.6f}"
    )

    percent_diff = (prob_binom - prob_trinom) / prob_trinom * 100.0

    print(f"\nDifference b/w trinom. and binom. wrt trinom.: {percent_diff:.6f} %")

    return (prob_binom, percent_diff)


def p_ml_from_g_pipeline(
    y1: int, y2: int, user_p: float
) -> tuple[ArgmaxG, float]:
    """
    Compute max. likelihood (ML) estimate value for p using g and compare
    with user-provided p. Return ML estimate result alongside percent difference
    wrt user-provided p
    """
    print("\n--- Finding max. likelihood estimate p from g(Y1 = y1|p) ---")

    ml_p_from_g = p_ml_argmax_prob_fn(y1, y2)
    ml_est_p = ml_p_from_g.max_likelihood_estimate_p
    
    print(f"ML estimate p from g(Y1 = y1|p) = {ml_est_p:.6f}")
    print("Converged?", ml_p_from_g.converged)
    print(f"Total optimization time: {ml_p_from_g.solver_time} s")

    percent_diff: float
    if user_p == 0.0:
        print("NOTICE: Percent difference is undefined for p=0.0, which you (yes you!) provided")
        percent_diff = float('nan')
    else:
        percent_diff = (ml_est_p - user_p) / user_p * 100.0

    print(
        f"Difference between ML estimate p and user-provided p: "
        f"{percent_diff:.6f} %"
    )
    print(
        f"Feature extractor probability from ML estimate p: {feature_extractor(y1, y2, ml_est_p):.6f}"
    )

    return (ml_p_from_g, percent_diff)


def main() -> None:
    print_usage()

    x = (
        int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    )
    p = float(sys.argv[4])

    validate_x_vector(x); validate_p_param(p)

    prob_trinom = trinomial_pipeline(x, p)

    y1 = x[0] + x[1]
    y2 = x[2]

    prob_binom, percent_diff_binom = feature_extractor_pipeline(y1, y2, p, prob_trinom)

    ml_p_from_g, percent_diff_ml_p_g = p_ml_from_g_pipeline(y1, y2, user_p=p)


if __name__ == "__main__":
    main()
