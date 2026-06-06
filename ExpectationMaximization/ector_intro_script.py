from probability.ector import trinomial_probability_det_objs, feature_extractor
from utils.validate import validate_x_vector, validate_p_param

import sys


def print_usage(
    usage: str = "python ector_intro_script.py x1 x2 x3 p"
) -> None:
    if len(sys.argv) != 5:
        print("Usage:", usage)

        sys.exit(0)


def main() -> None:
    print_usage()

    x = (
        int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    )
    p = float(sys.argv[4])

    validate_x_vector(x); validate_p_param(p)

    prob_trinom = trinomial_probability_det_objs(x, p)
    print(
        f"\nTrinom. probability of finding {x[0]} dark round objs,"
        f"{x[1]} dark square objs, {x[2]} light objs: {prob_trinom:.6f}"
    )

    prob_binom = feature_extractor(x, p)
    print(
        f"\nBinom. probability of finding {x[0]} dark round objs,"
        f"{x[1]} dark square objs, {x[2]} light objs: {prob_binom:.6f}"
    )

    print(f"\nDifference b/w trinom. and binom.: {abs(prob_trinom - prob_binom):.6f}")


if __name__ == "__main__":
    main()
