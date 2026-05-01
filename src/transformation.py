import numpy as np
import os
import csv
import math
import random

# ============================================================
# MODULE: Controlled Nonlinear Causal Rotation-Scaling Transform
#         with Optional Gaussian White Noise Augmentation
# ============================================================
#
# Author  : [Your Name / Team]
# Version : 2.0.0
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
# VERSION 2.0 ADDITIONS
# ---------------------
# Adds Gaussian white noise augmentation via multiplicative noise:
#
#     x_noisy = x · N(1, σ²),  clamped to [0.95|x|, 1.05|x|]
#
# This simulates realistic sensor/measurement noise while preserving:
#   - Signal polarity (sign of each element is never flipped)
#   - Approximate magnitude (bounded within ±5% of original)
#   - Statistical structure (noise is i.i.d. across elements)
#
# Four datasets are saved in total:
#   1. dataset_clean.csv       — original transform, no noise (CSV)
#   2. dataset_clean.npy       — original transform, no noise (NPY)
#   3. dataset_noisy.csv       — transform + Gaussian noise    (CSV)
#   4. dataset_noisy.npy       — transform + Gaussian noise    (NPY)
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
# NOTE: Invertibility holds only for CLEAN samples. Noisy samples
# cannot be exactly reconstructed due to the injected perturbations.
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

# Standard deviation for Gaussian white noise augmentation.
# Controls the spread of the multiplicative noise multiplier N(1, σ²).
# σ = 0.02 means ~95% of multipliers fall within [0.96, 1.04]
# before clamping, producing subtle, realistic perturbations.
_SIGMA = 0.02


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
# SECTION 3: GAUSSIAN WHITE NOISE FUNCTIONS
# ============================================================

def apply_noise_to_element(x: float, sigma: float) -> float:
    """
    Apply bounded multiplicative Gaussian white noise to a scalar.

    Noise is multiplicative rather than additive so that the
    perturbation magnitude scales with the signal magnitude.
    This prevents large absolute errors on small values and
    keeps signal-to-noise ratio (SNR) approximately constant
    across the dynamic range of the data.

    The noise multiplier is drawn from N(1, σ²) so that:
        E[x_noisy] = x · E[multiplier] = x · 1 = x
    i.e., the transform is unbiased in expectation.

    Clamping to ±5% of |x| truncates the Gaussian tails, turning
    the distribution into a truncated normal. This prevents extreme
    outliers that would arise from rare but large Gaussian draws,
    matching the behaviour of real bounded sensors.

    Sign is preserved explicitly to ensure noise never flips the
    polarity of a coordinate — a structural property of the data.

    Parameters
    ----------
    x : float
        Original scalar value. If x == 0, returns 0.0 immediately
        (no meaningful scale for multiplicative noise).
    sigma : float
        Standard deviation of the Gaussian multiplier N(1, σ²).
        Typical range: 0.01 (subtle) to 0.10 (strong).

    Returns
    -------
    float
        Noisy scalar with the same sign as x, magnitude bounded
        within [0.95·|x|, 1.05·|x|].

    Notes
    -----
    The ±5% hard clamp is independent of σ. For σ << 0.05 most
    draws are unaffected by the clamp; for σ >> 0.05 the clamp
    dominates and the distribution becomes approximately uniform
    on [0.95|x|, 1.05|x|].
    """
    # Edge case: zero cannot be scaled multiplicatively
    if x == 0.0:
        return 0.0

    # Draw multiplicative noise from N(1, σ²)
    # Mean = 1 preserves the expected value of the signal
    noise_multiplier = random.gauss(1, sigma)

    # Hard bounds: ±5% of the original magnitude
    # These are absolute limits regardless of the σ setting
    upper_bound = 1.05 * abs(x)
    lower_bound = 0.95 * abs(x)

    # Apply multiplier to magnitude (sign stripped to avoid
    # accidental polarity flip if multiplier is negative for σ >> 1)
    output = abs(x) * noise_multiplier

    # Clamp to safe bounds
    output = min(output, upper_bound)
    output = max(output, lower_bound)

    # Restore original sign — noise must never flip polarity
    output = float(np.sign(x) * output)

    return output


def apply_noise_to_vector(vec: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply independent Gaussian white noise to every element of a vector.

    Noise is applied element-wise and independently (i.i.d.), which
    is the defining property of white noise — zero correlation between
    any two components. This contrasts with coloured noise, where
    adjacent elements would be correlated.

    The anchor pair (indices 0 and 1) is intentionally included in
    the noise application here. Whether to exclude the anchor from
    noise (to preserve its exact role in decoding) is a design choice
    left to the caller; generate_dataset_noisy excludes the anchor
    for consistency with the clean transform's anchor-preservation
    guarantee.

    Parameters
    ----------
    vec : np.ndarray
        Flat 1-D array of real values. Shape is preserved.
    sigma : float
        Standard deviation passed to apply_noise_to_element.

    Returns
    -------
    np.ndarray
        New array of the same shape as vec with i.i.d. multiplicative
        Gaussian noise applied independently to each element.
    """
    return np.array([apply_noise_to_element(x, sigma) for x in vec])


# ============================================================
# SECTION 4: CORE TRANSFORMATION
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
# SECTION 5: DATASET GENERATION — CLEAN (ORIGINAL)
# ============================================================

def generate_dataset(v: np.ndarray, n_samples: int = 400) -> np.ndarray:
    """
    Generate a clean dataset of transformed outputs (no noise).

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
        No noise is applied; samples are exactly invertible.

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
# SECTION 6: DATASET GENERATION — NOISY (NEW)
# ============================================================

def generate_dataset_noisy(
    v: np.ndarray,
    n_samples: int = 400,
    sigma: float = _SIGMA
) -> np.ndarray:
    """
    Generate a noisy dataset of transformed outputs with Gaussian noise.

    Extends generate_dataset by injecting i.i.d. multiplicative
    Gaussian white noise into the non-anchor elements of each
    transformed sample. The anchor pair (indices 0–1) is kept
    noise-free to preserve its exact role as the geometric reference
    in any downstream decoding step.

    Noise model per non-anchor element x:
        x_noisy = sign(x) · clamp(|x| · N(1, σ²), 0.95|x|, 1.05|x|)

    This is multiplicative white noise — unbiased (E[x_noisy] = x),
    magnitude-proportional, polarity-preserving, and bounded.

    Parameters
    ----------
    v : np.ndarray
        Ground-truth input vector (flat, even-length).
    n_samples : int, optional
        Number of noisy output samples to generate. Default: 400.
    sigma : float, optional
        Standard deviation of the Gaussian noise multiplier N(1, σ²).
        Default: _SIGMA (module-level constant, 0.02).
        Higher σ → more aggressive noise; lower σ → subtler noise.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, len(v)); each row is a
        transformed + noisy version of v under a sampled λ.
        Anchor elements [0:2] are identical to the clean transform;
        all other elements carry bounded multiplicative noise.

    Notes
    -----
    Unlike the clean dataset, noisy samples are NOT exactly
    invertible. Applying transform(w_noisy, −λ) recovers an
    approximation of v, not v exactly. The reconstruction error
    is bounded by the clamp width (≤ 5% per element).

    Downstream decoding can still identify the correct λ by
    minimising intra-cluster variance across samples, but the
    reconstructed v will have small residual noise.
    """
    dataset = []

    for _ in range(n_samples):
        # Sample λ from the softmax-controlled distribution
        lam = np.random.choice(_LAMBDA_SET, p=_LAMBDA_PROBS)

        # Apply the clean geometric transform first
        w = transform(v, lam)

        # Inject i.i.d. Gaussian noise to non-anchor elements only.
        # Anchor (w[0:2]) is preserved to maintain its role as the
        # exact geometric reference for rotation/scale computation.
        w_noisy = w.copy()
        w_noisy[2:] = apply_noise_to_vector(w[2:], sigma)

        dataset.append(w_noisy)

    return np.array(dataset)


# ============================================================
# SECTION 7: ENTRY POINT — USER INPUT
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
# SECTION 8: GENERATE BOTH DATASETS
# ============================================================

print("\nGenerating clean dataset (400 samples, no noise)...")
dataset_clean = generate_dataset(v, n_samples=400)

print("Generating noisy dataset (400 samples, σ={:.3f})...".format(_SIGMA))
dataset_noisy = generate_dataset_noisy(v, n_samples=400, sigma=_SIGMA)


# ============================================================
# SECTION 9: REVERSIBILITY VERIFICATION (CLEAN ONLY)
# ============================================================

# Noise breaks exact invertibility, so verification applies
# only to the clean transform. Noisy reconstruction is approximate.

print("\n--- Reversibility Check (Clean Transform) ---")
print(f"{'λ':>8}  {'Max Reconstruction Error':>26}  {'Pass':>6}")
print("-" * 48)

for lam in _LAMBDA_SET:
    w     = transform(v, lam)
    v_rec = transform(w, -lam)
    error = np.max(np.abs(v - v_rec))
    passed = np.allclose(v, v_rec)

    print(f"{lam:>+8.2f}  {error:>26.2e}  {'✓' if passed else '✗':>6}")


# ============================================================
# SECTION 10: APPROXIMATE RECONSTRUCTION CHECK (NOISY)
# ============================================================

# Demonstrates that noisy samples reconstruct v approximately,
# with bounded error proportional to the noise clamp width.

print("\n--- Approximate Reconstruction Check (Noisy Transform) ---")
print(f"{'λ':>8}  {'Mean Reconstruction Error':>26}  {'Max Error':>12}")
print("-" * 55)

for lam in _LAMBDA_SET:
    # Generate a single noisy sample for this λ
    w_clean = transform(v, lam)
    w_noisy = w_clean.copy()
    w_noisy[2:] = apply_noise_to_vector(w_clean[2:], _SIGMA)

    # Attempt inverse: will not be exact due to noise
    v_approx = transform(w_noisy, -lam)
    errors = np.abs(v - v_approx)
    mean_err = np.mean(errors)
    max_err  = np.max(errors)

    print(f"{lam:>+8.2f}  {mean_err:>26.2e}  {max_err:>12.2e}")


# ============================================================
# SECTION 11: SAVE ALL FOUR DATASETS TO DISK
# ============================================================

_SAVE_DIR = r"C:\Users\DELL\Documents\GitHub\vector_Transformations\data"
os.makedirs(_SAVE_DIR, exist_ok=True)

# Dataset 1 & 2: Clean (CSV + NPY)
_csv_clean = os.path.join(_SAVE_DIR, "dataset_clean.csv")
_npy_clean = os.path.join(_SAVE_DIR, "dataset_clean.npy")
np.savetxt(_csv_clean, dataset_clean, delimiter=",")
np.save(_npy_clean, dataset_clean)

# Dataset 3 & 4: Noisy (CSV + NPY)
_csv_noisy = os.path.join(_SAVE_DIR, "dataset_noisy.csv")
_npy_noisy = os.path.join(_SAVE_DIR, "dataset_noisy.npy")
np.savetxt(_csv_noisy, dataset_noisy, delimiter=",")
np.save(_npy_noisy, dataset_noisy)

print(f"\nAll four datasets saved to: {_SAVE_DIR}")
print(f"  Clean CSV → {_csv_clean}")
print(f"  Clean NPY → {_npy_clean}")
print(f"  Noisy CSV → {_csv_noisy}")
print(f"  Noisy NPY → {_npy_noisy}")


# ============================================================
# SECTION 12: DISPLAY SAMPLE OUTPUT
# ============================================================

print("\nFirst 5 CLEAN samples (rounded to 3 d.p.):")
print(np.round(dataset_clean[:5], 3))

print("\nFirst 5 NOISY samples (rounded to 3 d.p.):")
print(np.round(dataset_noisy[:5], 3))

print("\nElement-wise noise magnitude (sample 0, non-anchor elements):")
diff = np.abs(dataset_clean[0, 2:] - dataset_noisy[0, 2:])
print(f"  Max delta: {diff.max():.4f},  Mean delta: {diff.mean():.4f}")


# ============================================================
# SECTION 13: EMPIRICAL λ SAMPLING FREQUENCIES
# ============================================================

print("\nEmpirical λ sampling frequencies — CLEAN dataset (≈ softmax probs):")
for lam in _LAMBDA_SET:
    reconstructed = np.array([transform(w, -lam) for w in dataset_clean])
    matches = np.sum([np.allclose(r, v) for r in reconstructed])
    print(f"  λ = {lam:+.2f} : {matches:>4} / {len(dataset_clean)} samples "
          f"  (expected ~{_LAMBDA_PROBS[np.where(_LAMBDA_SET == lam)[0][0]]:.1%})")

print("\nEmpirical λ sampling frequencies — NOISY dataset (approximate, atol=0.1):")
for lam in _LAMBDA_SET:
    reconstructed = np.array([transform(w, -lam) for w in dataset_noisy])
    # Use relaxed tolerance because noise breaks exact allclose
    matches = np.sum([np.allclose(r, v, atol=0.1) for r in reconstructed])
    print(f"  λ = {lam:+.2f} : {matches:>4} / {len(dataset_noisy)} samples "
          f"  (expected ~{_LAMBDA_PROBS[np.where(_LAMBDA_SET == lam)[0][0]]:.1%})")