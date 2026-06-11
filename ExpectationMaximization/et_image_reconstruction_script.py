from ETImageReconstruction.objs.body import Body
from ETImageReconstruction.objs.box import Box
from ETImageReconstruction.objs.detector import Detector
from utils.plotters.et_image_reconstruction_plots import plot_poisson_pmf

import sys
import numpy as np
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent


def print_usage() -> None:
    if len(sys.argv) != 3:
        usage = "python et_image_reconstruction_script.py num_boxes num_detectors"
        print("Usage:", usage)

        sys.exit(0)


def et_image_reconstruction_pipeline(
    num_boxes: int, num_detectors: int
) -> None:
    boxes = [Box() for _ in range(num_boxes)]
    detectors = [Detector() for _ in range(num_detectors)]

    body = Body(detectors, boxes)
    
    body.compute_lambda_d()
    lambda_d = np.array([
        detector.lambda_d for detector in body.detectors
    ], dtype=np.float64)

    body.simulate_detector_counts()

    poisson_pmf = body.poisson_pmf_counts()

    plot_poisson_pmf(
        poisson_pmf, lambda_d,
        PROJECT_ROOT / "plots"
    )
    

def main() -> None:
    print_usage()

    num_boxes = int(sys.argv[1])
    num_detectors = int(sys.argv[2])

    et_image_reconstruction_pipeline(num_boxes, num_detectors)


if __name__ == "__main__":
    main()
