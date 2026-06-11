# sigint
Code to model several white papers related to signal intelligence.


# Expectation-Maximization Algorithm — Signal Processing Implementation

Python implementation of the EM algorithm based on:

> Moon, T.K. (1996). "The Expectation-Maximization Algorithm." *IEEE Signal Processing Magazine*, 13(6), 47–60.

Currently implements Ector's Problem as an introductory example.

---

## Ector's Problem

Ector's Problem models an image pattern-recognition scenario with three classes of objects:

- **Round dark objects** (X1): fixed class probability 1/4
- **Square dark objects** (X2): class probability 1/4 + p/4
- **Light objects** (X3): class probability 1/2 - p/4

The unknown scalar parameter `p` governs the distribution. A feature extractor observes only the total dark count (y1 = x1 + x2) and light count (y2 = x3), unable to distinguish round from square. The EM algorithm estimates the underlying x1 and x2 counts and converges on a maximum-likelihood (ML) estimate of `p`.

### Implemented Equations

- **Equations (1) & (2):** Trinomial PMF
- **Equation (3):** ML estimate of p via argmax of g(Y1=y1|p)
- **Equations (5) & (6):** E-step — conditional expectations for x1 and x2
- **Equation (7):** M-step — parameter update for p
- **Box 2:** Multinomial combination and conditional expectations

---

## Usage

```
python ector_intro_script.py x1 x2 x3 p
```

| Argument | Type | Description |
|----------|------|-------------|
| `x1` | non-negative int | True count of round dark objects |
| `x2` | non-negative int | True count of square dark objects |
| `x3` | non-negative int | True count of light objects |
| `p` | float in [-1, 2] | Warm-start value for EM parameter estimation |

### Example

```
python ector_intro_script.py 25 38 37 0
```

---

## Implementation Notes

- The EM implementation follows Equations (5), (6), and (7) separately as well as the combined one-step update in Equation (8).
- `x3 = 0` is not supported — the M-step denominator collapses to zero in this case.
- The paper's numerical example states n=100 and y1=100 with x1=25 and x2=38, which is inconsistent since x1+x2=63≠100. This implementation uses x3=37 (so that n=100) as the most plausible interpretation.

---

## Limitations

- Small sample sizes produce unreliable parameter estimates. Larger n yields estimates that converge closer to the true p.
- The ML estimate of p from the feature extractor (Equation 3) is "optimal" only with respect to the coarser binomial observation (y1, y2), not the full trinomial (x1, x2, x3).

---

# ET Image Reconstruction
Models emission tomography (ET) image reconstruction, where tissues emit photons detected by surrounding sensors. The body is divided into `B` boxes, each with a Poisson emission process with mean `λ(b)`. `D` detectors surrounding the body observe photon counts `y(d)`.
Structure

Box: owns emission rate `λ(b)`, randomly initialized via exponential distribution
Detector: owns observed photon count `y(d)` and mean detection rate `λ(d)`
Body: owns boxes, detectors, and detection probability matrix `p(b,d)` (num_boxes × num_detectors, rows sum to 1 per **Equation 14**)

## Implemented Equations

- **Equation (15)**: λ(b,d) = λ(b) · p(b,d)
- **Poisson PMF: f(n|λ(b))**: photon generation process per box
- **f(y|λ(d))**: Poisson PMF for detector counts (incomplete data likelihood)
- **λ(d) computation**: mean photon count per detector from box emission rates

---

## Usage

```
python et_image_reconstruction_script.py num_boxes num_detectors
```

| Argument | Type | Description | 
| -------- | ---- | ----------- | 
| num_boxes | positive int | Number of boxes dividing the body |
| num_detectors | positive int| Number of detectors surrounding the body |

### Example
python et_image_reconstruction_script.py 20 50
Outputs a bar chart of Poisson PMF values overlaid with λ(d) across all detectors, saved as poisson_pmf.png.

---

## Planned Extensions

- Active noise cancellation (ANC)
- Hidden Markov models (HMMs)
- Spread-spectrum multi-user communication

---

## Requirements

- Python 3.10+
- numpy
- scipy
- matplotlib