import numpy as np
import numpy.typing as npt
import math

from .box import Box
from .detector import Detector
from utils.validate import validate_photon_counts


class Body:
    def __init__(
        self,
        detectors: list[Detector],
        boxes: list[Box]
    ) -> None:
        self.detectors = detectors
        self.boxes = boxes

        self.num_detectors = len(detectors)
        self.num_boxes = len(boxes)

        self.p_bd = np.random.dirichlet(
            alpha=np.ones(self.num_detectors), size=self.num_boxes
        )


    def photon_generation_process(
        self, photon_counts: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """
        The generation of photons can be described as
        a Poisson process.
        """
        validate_photon_counts(photon_counts, self.boxes)

        photon_generation_arr = np.empty(shape=self.num_boxes, dtype=np.float64)

        # TODO: determine if it is better to foresake the Box class
        # so array broadcasting can be done here
        for i, box in enumerate(self.boxes):
            photon_generation_arr[i] = (
                np.exp(-box.lambda_b) * (
                    box.lambda_b ** photon_counts[i] / math.factorial(photon_counts[i])
                )
            )

        return photon_generation_arr
    

    def compute_lambda_d(self) -> None:
        """
        Compute mean photon count for each detector from box
        emission rates and detection probabilities
        """
        for d, detector in enumerate(self.detectors):
            detector.lambda_d = float(sum(
                box.lambda_b * self.p_bd[b, d]
                for b, box in enumerate(self.boxes)
            ))


    def simulate_detector_counts(self) -> None:
        """
        Draw observed photon counts y(d) from Poisson(lambda_d)
        for each detector
        """
        for detector in self.detectors:
            detector.y = np.random.poisson(lam=detector.lambda_d)

    
    def poisson_pmf_counts(self) -> npt.NDArray[np.float64]:
        """
        Compute the Poisson PMF for detector counts
        """
        poisson_pmf = np.empty(self.num_detectors, dtype=np.float64)

        for i, detector in enumerate(self.detectors):
            poisson_pmf[i] = np.exp(-detector.lambda_d) * (
                detector.lambda_d ** detector.y / math.factorial(detector.y)
            )

        return poisson_pmf
