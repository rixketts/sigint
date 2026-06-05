from probability.ector import trinomial_probability_det_objs

import sys


def print_usage(
    usage: str = "python ector_intro_script.py x1 x2 x3 p"
) -> None:
    if len(sys.argv) != 5:
        print("Usage:", usage)

        sys.exit(0)


def main() -> None:
    print_usage()

    x1 = int(sys.argv[1])
    x2 = int(sys.argv[2])
    x3 = int(sys.argv[3])

    p = float(sys.argv[4])

    prob = trinomial_probability_det_objs(x=(x1, x2, x3), p=p)

    print(f"\nProbability of finding {x1} dark round objs, {x2} dark square objs, {x3} light objs: {prob:.6f}")


if __name__ == "__main__":
    main()
