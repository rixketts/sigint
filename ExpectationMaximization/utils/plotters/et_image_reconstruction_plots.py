import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pathlib import Path


def plot_poisson_pmf(
    poisson_pmf: npt.NDArray[np.float64],
    lambda_d: npt.NDArray[np.float64],
    output_path: Path
) -> None:
    """
    Plot Poisson PMF values and mean detection rates across detectors
    """
    detector_indices = np.arange(len(poisson_pmf))

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.bar(
        x=detector_indices, height=poisson_pmf, color="steelblue",
        alpha=0.7, label="Poisson PMF"
    )
    ax1.set_xlabel("Detector index")
    ax1.set_ylabel("Poisson PMF", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(
        detector_indices, lambda_d, color="darkorange", 
        marker="o", linewidth=1.5, label="λ(d)"
    )
    ax2.set_ylabel("λ(d)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    fig.suptitle("Poisson PMF and Mean Detection Rates Across Detectors")
    fig.legend(
        loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes
    )

    plt.tight_layout()

    plt.savefig(
        output_path / "poisson_pmf.png", dpi=150, bbox_inches="tight"
    )

    plt.close()
