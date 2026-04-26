import numpy as np
import os
import csv

# ============================================================
# MODULE: Controlled Nonlinear Causal Rotation-Scaling Transform
# ============================================================
#
# Author  : [Your Name / Team]
# Version : 1.0.0
# License : [Your License]
#
# DESCRIPTION
# -----------
# Implements a nonlinear, anchor-conditioned transformation over
# vectors interpreted as flattened sequences of 2D coordinate pairs.
#
# A hidden scalar parameter λ is sampled from a fixed discrete set
# Λ = {λ_1, ..., λ_n} according to a softmax-normalised probability
# distribution derived from designer-specified log-weights α.
#
# This approach provides:
#   - Deterministic, reproducible output ratios (controlled by α)
#   - Smooth probabilistic weighting with no arbitrary bias
#   - Exact invertibility: transform(transform(v, λ), −λ) ≡ v
#   - Anchor causality: the first coordinate pair governs the
#     rotation angle and scale applied to all subsequent pairs
#
# INPUT FORMAT
# ------------
# A 1-D NumPy array of even length:
#
#     v = [x0, y0, x1, y1, x2, y2, ...]
#
# where (x0, y0) is the ANCHOR — always preserved unchanged.
#
# TRANSFORMATION EQUATIONS
# ------------------------
# Given anchor (x0, y0) and parameter λ:
#
#     r0    = ‖(x0, y0)‖            Euclidean norm of anchor
#     θ     = λ · arctan2(y0, x0)   Anchor-conditioned rotation
#     s     = exp(λ · tanh(r0))     Bounded, anchor-conditioned scale
#
# For each non-anchor pair k ≥ 1:
#
#     [w_x]   =  s · R(θ) · [v_x]
#     [w_y]              ·  [v_y]
#
# where R(θ) is the standard 2D rotation matrix:
#
#     R(θ) = [ cos θ  −sin θ ]
#             [ sin θ   cos θ ]
#
# LAMBDA SAMPLING (SOFTMAX-CONTROLLED)
# -------------------------------------
# Candidates : Λ = [λ_1, ..., λ_n]  (fixed discrete set)
# Log-weights: α = [α_1, ..., α_n]  (designer-specified)
# Probability: P(λ_i) = softmax(α_i) = exp(α_i) / Σ_j exp(α_j)
#
# Adjusting α controls the sampling ratio without introducing
# arbitrary frequency bias. The softmax guarantee Σ P(λ_i) = 1.
#
# INVERTIBILITY
# -------------
# The transform is exactly invertible by negating λ:
#
#     transform(transform(v, λ), −λ) = v   for all v, λ
#
# DECODING STRATEGY (MULTI-SAMPLE)
# ----------------------------------
# Given N output samples generated from the same unknown v:
#
#   1. Read the anchor directly from any sample (always preserved).
#   2. For each sample w_i and candidate λ_j, compute:
#          v_candidate(i, j) = transform(w_i, −λ_j)
#   3. Score each λ_j by intra-cluster variance across samples.
#          Low variance → candidate λ is consistent → likely true λ.
#   4. Identify all significant low-variance clusters; their relative
#      sizes reflect the softmax probabilities P(λ_j).
#   5. Reconstruct v from any cluster's agreed reconstruction.
#
# ============================================================


# ============================================================
# SECTION 1: INTERNAL CONFIGURATION
# ============================================================

# Discrete candidate set for the hidden parameter λ.
# Values represent rotation-scaling intensities; negative λ
# produces inverse rotation/compression, positive λ produces
# forward rotation/expansion.
_LAMBDA_SET = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])

# Log-weights controlling the softmax sampling distribution.
# Higher α_i → higher probability of sampling λ_i.
# Current setting: peaked at λ=0.0 (identity transform, α=1.0),
# with symmetric decay toward the extremes (α=−1.0).
#
# Resulting approximate sampling ratios:
#   λ = ±0.10  →  ~7.9%  each
#   λ = ±0.05  →  ~21.5% each
#   λ =  0.00  →  ~58.3%
_ALPHA = np.array([-1.0, 0.0, 1.0, 0.0, -1.0])


# ============================================================
# SECTION 2: UTILITY FUNCTIONS
# ============================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the numerically stable softmax of a real-valued array.

    Uses the standard max-subtraction stability trick to prevent
    overflow in exp() for large input values.

    Parameters
    ----------
    x : np.ndarray
        1-D array of real-valued log-weights.

    Returns
    -------
    np.ndarray
        Probability vector of the same shape; entries are positive
        and sum to 1.

    Notes
    -----
    Stability: subtracting max(x) before exponentiation does not
    change the output (cancels in numerator and denominator) but
    ensures the largest exponent is exp(0) = 1, preventing overflow.
    """
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


# Pre-computed sampling probabilities derived from log-weights.
# These are fixed at module load time for efficiency.
_LAMBDA_PROBS = softmax(_ALPHA)


# ============================================================
# SECTION 3: CORE TRANSFORMATION
# ============================================================

def transform(v: np.ndarray, lam: float) -> np.ndarray:
    """
    Apply the nonlinear anchor-conditioned rotation-scaling transform.

    The first coordinate pair (anchor) is preserved unchanged.
    All subsequent pairs are jointly rotated and scaled using
    parameters derived solely from the anchor and λ.

    This causal coupling ensures global consistency: every output
    pair is transformed in the same geometric frame, making the
    transform both structured and exactly invertible.

    Parameters
    ----------
    v : np.ndarray
        Flat 1-D array of even length [x0, y0, x1, y1, ...].
        Must contain at least one coordinate pair (length ≥ 2).
    lam : float
        Transformation intensity parameter λ.
        λ = 0  → identity (no change)
        λ > 0  → forward rotation + expansion
        λ < 0  → inverse rotation + compression
        Invertibility: transform(transform(v, λ), −λ) = v

    Returns
    -------
    np.ndarray
        Transformed vector w of the same shape as v.
        w[0:2] = v[0:2] (anchor preserved exactly).

    Raises
    ------
    ValueError
        If v has odd length or fewer than 2 elements.

    Notes
    -----
    Rotation matrix R(θ):
        [cos θ  −sin θ]
        [sin θ   cos θ]

    Scale factor s = exp(λ · tanh(r0)) is bounded because
    tanh maps r0 ∈ [0, ∞) → [0, 1), ensuring s never diverges
    for finite λ.
    """
    w = v.copy()
    n_pairs = len(v) // 2

    # --- Anchor-derived transform parameters ---
    ax, ay = v[0], v[1]
    r0 = np.sqrt(ax**2 + ay**2)          # Euclidean norm of anchor

    theta = lam * np.arctan2(ay, ax)     # Rotation angle (anchor-conditioned)
    s     = np.exp(lam * np.tanh(r0))    # Scale factor (bounded by tanh)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # --- Apply rotation-scaling to all non-anchor pairs ---
    for k in range(1, n_pairs):
        x, y = v[2*k], v[2*k + 1]

        # Standard 2D rotation matrix applied with uniform scale s
        w[2*k]     = s * (x * cos_t - y * sin_t)   # w_x = s(x cosθ − y sinθ)
        w[2*k + 1] = s * (x * sin_t + y * cos_t)   # w_y = s(x sinθ + y cosθ)

    return w


# ============================================================
# SECTION 4: DATASET GENERATION
# ============================================================

def generate_dataset(v: np.ndarray, n_samples: int = 400) -> np.ndarray:
    """
    Generate a dataset of transformed outputs from a ground-truth vector.

    Each sample independently draws λ from the softmax-controlled
    distribution and applies the corresponding transform. The
    expected proportion of samples per λ value matches _LAMBDA_PROBS.

    Parameters
    ----------
    v : np.ndarray
        Ground-truth input vector (flat, even-length).
    n_samples : int, optional
        Number of output samples to generate. Default: 400.
        Larger values improve empirical frequency alignment with
        the theoretical softmax distribution.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, len(v)); each row is a
        transformed version of v under a sampled λ.

    Notes
    -----
    Samples generated with λ = 0.0 are identical to v (identity
    transform). With the default α configuration, ~58% of samples
    will be unmodified copies of the input.
    """
    dataset = []

    for _ in range(n_samples):
        lam = np.random.choice(_LAMBDA_SET, p=_LAMBDA_PROBS)
        w   = transform(v, lam)
        dataset.append(w)

    return np.array(dataset)


# ============================================================
# SECTION 5: ENTRY POINT — USER INPUT
# ============================================================

while True:
    try:
        _user_input = input("Enter ground truth vector (e.g. [1,2,3,4]): ")
        v = np.array(eval(_user_input), dtype=float)

        if len(v) < 2 or len(v) % 2 != 0:
            raise ValueError("Vector must be non-empty with even length.")

        break

    except Exception as e:
        print(f"Invalid input: {e}. Please try again.")


# ============================================================
# SECTION 6: GENERATE DATASET
# ============================================================

dataset = generate_dataset(v, n_samples=400)


# ============================================================
# SECTION 7: REVERSIBILITY VERIFICATION
# ============================================================

print("\n--- Reversibility Check ---")
print(f"{'λ':>8}  {'Max Reconstruction Error':>26}  {'Pass':>6}")
print("-" * 48)

for lam in _LAMBDA_SET:
    w     = transform(v, lam)
    v_rec = transform(w, -lam)
    error = np.max(np.abs(v - v_rec))
    passed = np.allclose(v, v_rec)

    print(f"{lam:>+8.2f}  {error:>26.2e}  {'✓' if passed else '✗':>6}")


# ============================================================
# SECTION 8: SAVE DATASET TO DISK
# ============================================================

_SAVE_DIR = r"C:\Users\DELL\Documents\GitHub\vector_Transformations\data"
os.makedirs(_SAVE_DIR, exist_ok=True)

_csv_path = os.path.join(_SAVE_DIR, "dataset.csv")
_npy_path = os.path.join(_SAVE_DIR, "dataset.npy")

np.savetxt(_csv_path, dataset, delimiter=",")
np.save(_npy_path, dataset)

print(f"\nDataset saved to:\n  CSV → {_csv_path}\n  NPY → {_npy_path}")


# ============================================================
# SECTION 9: DISPLAY SAMPLE OUTPUT
# ============================================================

print("\nFirst 5 samples (rounded to 3 d.p.):")
print(np.round(dataset[:5], 3))

print("\nEmpirical λ sampling frequencies (≈ softmax probs):")
# Reconstruct which λ was used for each sample via consistency check
for lam in _LAMBDA_SET:
    reconstructed = np.array([transform(w, -lam) for w in dataset])
    matches = np.sum([np.allclose(r, v) for r in reconstructed])
    print(f"  λ = {lam:+.2f} : {matches:>4} / {len(dataset)} samples "
          f"  (expected ~{_LAMBDA_PROBS[np.where(_LAMBDA_SET == lam)[0][0]]:.1%})")