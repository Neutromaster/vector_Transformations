"""
============================================================
MODULE: Controlled Nonlinear Causal Rotation-Scaling Transform
        with Softmax-Weighted Sampling and Gaussian Noise Augmentation
============================================================

Author  : [Your Name / Team]
Version : 3.0.0
License : [Your License]

OVERVIEW
--------
This module implements a nonlinear, anchor-conditioned geometric
transformation over vectors interpreted as flattened sequences of
2D coordinate pairs.  It is structured around three clearly separated
dataset-generation modes, each building on the previous:

    1. generate_dataset_raw(v, n_samples)
       ─────────────────────────────────
       Samples λ UNIFORMLY at random from the candidate set Λ and
       applies the clean transform.  No softmax weighting, no noise.
       Useful as a baseline / ablation: every λ appears with equal
       probability regardless of the α log-weight configuration.

    2. generate_dataset_weighted(v, n_samples)
       ────────────────────────────────────────
       Samples λ from the SOFTMAX distribution derived from α.
       Still a clean (noise-free) transform; samples are exactly
       invertible.  This is the "production" clean dataset.

    3. generate_dataset_noisy(v, n_samples, sigma)
       ─────────────────────────────────────────────
       Same softmax-weighted sampling as (2), then injects i.i.d.
       bounded multiplicative Gaussian noise into every non-anchor
       element.  The anchor pair is kept exact so that downstream
       decoders can still compute the geometric reference frame.

TRANSFORMATION EQUATIONS
------------------------
Given anchor (x₀, y₀) and parameter λ:

    r₀  = ‖(x₀, y₀)‖              Euclidean norm of anchor
    θ   = λ · arctan2(y₀, x₀)    Anchor-conditioned rotation angle
    s   = exp(λ · tanh(r₀))      Bounded, anchor-conditioned scale

For each non-anchor pair k ≥ 1:

    [w_x]   = s · R(θ) · [v_x]
    [w_y]                [v_y]

where R(θ) is the standard 2-D rotation matrix:

    R(θ) = [ cos θ  −sin θ ]
            [ sin θ   cos θ ]

LAMBDA SAMPLING
---------------
Candidate set : Λ = [λ₁, …, λₙ]
Log-weights   : α = [α₁, …, αₙ]   (designer-specified)
Uniform prob  : P_raw(λᵢ)  = 1/n
Softmax prob  : P_soft(λᵢ) = exp(αᵢ) / Σⱼ exp(αⱼ)

NOISE MODEL (dataset 3 only)
-----------------------------
Per non-anchor element x:

    x_noisy = sign(x) · clamp(|x| · N(1, σ²),  0.95|x|,  1.05|x|)

Properties:
  · Unbiased         : E[x_noisy] = x
  · Magnitude-scaled : noise grows with signal magnitude (constant SNR)
  · Polarity-safe    : sign is never flipped
  · Bounded          : absolute error ≤ 5 % of |x|

INVERTIBILITY
-------------
    transform(transform(v, λ), −λ) = v   ∀ v, λ

Holds exactly for datasets (1) and (2).  Dataset (3) admits only
approximate recovery; max element-wise error ≤ 5 % of |x|.

OUTPUT FILES
------------
Six files are written to _SAVE_DIR:

    dataset_raw.csv       / dataset_raw.npy
    dataset_weighted.csv  / dataset_weighted.npy
    dataset_noisy.csv     / dataset_noisy.npy
============================================================
"""

import math
import os
import random

import numpy as np


# ============================================================
# SECTION 1: MODULE-LEVEL CONFIGURATION
# ============================================================

# Discrete candidate set for the hidden parameter λ.
# Negative values → inverse rotation + compression.
# Positive values → forward rotation + expansion.
# Zero            → identity (no change to v).
_LAMBDA_SET: np.ndarray = np.array([-0.10, -0.05, 0.00, 0.05, 0.10])

# Designer-specified log-weights that control the softmax sampling
# distribution.  Higher αᵢ → higher probability of drawing λᵢ.
#
# Current setting (peaked at λ = 0.00, α = 1.0):
#   λ = ±0.10  →  ~ 7.9 %  each
#   λ = ±0.05  →  ~21.5 %  each
#   λ =  0.00  →  ~58.3 %
_ALPHA: np.ndarray = np.array([-1.0, 0.0, 1.0, 0.0, -1.0])

# Standard deviation of the multiplicative Gaussian noise multiplier.
# σ = 0.02 → ~95 % of draws land in [0.96, 1.04] before clamping.
_SIGMA: float = 0.02

# Output directory for all saved datasets.
_SAVE_DIR: str = r"C:\Users\DELL\Documents\GitHub\vector_Transformations\data"


# ============================================================
# SECTION 2: UTILITY — SOFTMAX
# ============================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the numerically stable softmax of a 1-D array.

    The max-subtraction trick is applied before exponentiation:

        softmax(x)ᵢ = exp(xᵢ − max(x)) / Σⱼ exp(xⱼ − max(x))

    Subtracting max(x) does not change the result mathematically
    (it cancels in numerator and denominator) but guarantees that
    the largest exponent evaluated is exp(0) = 1, preventing
    numerical overflow for large inputs.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Real-valued log-weights.  May contain negative values,
        zeros, or positives in any order.

    Returns
    -------
    np.ndarray, shape (n,)
        Probability vector.  All entries are strictly positive
        and sum to exactly 1 (up to floating-point rounding).

    Examples
    --------
    >>> softmax(np.array([1.0, 0.0, -1.0]))
    array([0.665, 0.245, 0.090])   # approx
    """
    shifted = x - np.max(x)            # stability: shift so max exp arg = 0
    exp_x   = np.exp(shifted)          # element-wise exponentiation
    return exp_x / np.sum(exp_x)       # normalise to sum = 1


# Pre-compute the softmax probabilities once at module load time.
# Recomputing them per call would waste cycles; they are constant
# as long as _ALPHA and _LAMBDA_SET do not change at runtime.
_LAMBDA_PROBS: np.ndarray = softmax(_ALPHA)


# ============================================================
# SECTION 3: UTILITY — GAUSSIAN NOISE
# ============================================================

def _apply_noise_to_scalar(x: float, sigma: float) -> float:
    """
    Inject bounded multiplicative Gaussian white noise into a scalar.

    Noise is multiplicative (not additive) so that the perturbation
    magnitude scales linearly with the signal magnitude, keeping the
    signal-to-noise ratio (SNR) approximately constant across the
    full dynamic range of the data.

    Noise multiplier M ~ N(1, σ²):

        x_noisy = |x| · M           (raw noisy magnitude)
        x_noisy = clamp(x_noisy, 0.95|x|, 1.05|x|)   (bounded)
        x_noisy = sign(x) · x_noisy                   (polarity restored)

    Why clamp?
    ----------
    A Gaussian distribution has unbounded tails; without clamping,
    rare large draws would produce arbitrarily large perturbations.
    The ±5 % hard cap matches the behaviour of real bounded sensors
    and keeps reconstruction error within a known, controlled range.

    Why preserve sign?
    ------------------
    The coordinate data has geometric meaning; flipping the sign of
    a coordinate would produce a reflection, not a small perturbation.
    Sign preservation ensures the noisy sample stays in the same
    geometric neighbourhood as the clean sample.

    Why return 0 immediately for x == 0?
    -------------------------------------
    Multiplicative noise on zero produces zero regardless of M,
    so the clamp/sign logic would all reduce to 0.0 anyway.
    The early return avoids division by zero in any potential
    SNR-based analysis and is semantically cleaner.

    Parameters
    ----------
    x : float
        Original scalar.  May be positive, negative, or zero.
    sigma : float
        Standard deviation of N(1, σ²).  Recommended range:
        0.01 (very subtle) to 0.10 (strongly noisy).
        Values above ~0.05 cause the hard clamp to dominate.

    Returns
    -------
    float
        Noisy scalar with the same sign as x and magnitude in
        [0.95·|x|,  1.05·|x|].

    Notes
    -----
    The ±5 % clamp width is a deliberate design constant independent
    of σ.  For σ ≪ 0.05 most draws are within bounds naturally;
    for σ ≫ 0.05 the distribution on the output approaches
    approximately Uniform(0.95|x|, 1.05|x|).
    """
    # Edge case: zero cannot be meaningfully perturbed multiplicatively.
    if x == 0.0:
        return 0.0

    abs_x = abs(x)

    # Draw multiplier from N(1, σ²); mean = 1 keeps the transform unbiased.
    multiplier = random.gauss(1.0, sigma)

    # Compute raw noisy magnitude (strip sign before clamping to avoid
    # accidental polarity flip when multiplier is negative, which can
    # happen if σ is large enough).
    raw = abs_x * multiplier

    # Hard clamp: magnitude must stay within ±5 % of original.
    raw = min(raw, 1.05 * abs_x)
    raw = max(raw, 0.95 * abs_x)

    # Restore original polarity.
    return float(math.copysign(raw, x))


def _apply_noise_to_vector(vec: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply i.i.d. Gaussian white noise to every element of a 1-D vector.

    "White" noise means zero cross-element correlation: each element
    is perturbed independently with its own draw from N(1, σ²).
    This contrasts with coloured noise (e.g., Brownian), where adjacent
    elements would be correlated.

    The function is a thin vectorised wrapper around
    `_apply_noise_to_scalar`.  It is intentionally kept separate so
    that element-level noise logic remains testable in isolation.

    Parameters
    ----------
    vec : np.ndarray, shape (m,)
        Flat 1-D array of real values.  Any finite floats are accepted.
    sigma : float
        Standard deviation forwarded to `_apply_noise_to_scalar`.

    Returns
    -------
    np.ndarray, shape (m,)
        New array of the same shape; each element independently noised.
        The input `vec` is not modified in place.
    """
    return np.array([_apply_noise_to_scalar(float(x), sigma) for x in vec])


# ============================================================
# SECTION 4: CORE TRANSFORMATION
# ============================================================

def transform(v: np.ndarray, lam: float) -> np.ndarray:
    """
    Apply the nonlinear anchor-conditioned rotation-scaling transform.

    GEOMETRY
    --------
    The first coordinate pair (v[0], v[1]) is the ANCHOR and is
    carried unchanged into the output.  It defines the global
    geometric reference frame: the rotation angle θ and scale
    factor s are derived exclusively from the anchor and λ.

    All subsequent pairs (v[2k], v[2k+1]) for k = 1, 2, … are
    jointly transformed by the same (θ, s), ensuring every output
    pair lives in a consistent geometric frame.

    PARAMETER DERIVATION
    --------------------
        ax, ay   =  v[0], v[1]
        r₀       =  √(ax² + ay²)          Euclidean norm of anchor
        θ        =  λ · arctan2(ay, ax)   Rotation angle
        s        =  exp(λ · tanh(r₀))     Scale factor

    Why tanh(r₀)?
    The scale factor must be bounded for the transform to be
    numerically stable and practically useful.  tanh maps
    r₀ ∈ [0, ∞) → [0, 1), so s = exp(λ · tanh(r₀)) is always
    finite regardless of how large r₀ or λ become.

    ROTATION MATRIX
    ---------------
        R(θ) = [ cos θ   −sin θ ]
               [ sin θ    cos θ ]

    Applied as:
        w_x = s · (x · cos θ − y · sin θ)
        w_y = s · (x · sin θ + y · cos θ)

    INVERTIBILITY
    -------------
    Negating λ exactly reverses the transform:

        transform(transform(v, λ), −λ) = v   ∀ v, λ

    Proof sketch:
    - θ' = −λ · arctan2(ay, ax) = −θ   → R(−θ) = R(θ)ᵀ
    - s' = exp(−λ · tanh(r₀))  = 1/s   → s' · s = 1
    - Applying s'·R(−θ) to s·R(θ)·p yields s'·s · R(−θ)·R(θ)·p = I·p = p.

    Parameters
    ----------
    v : np.ndarray, shape (2n,),  n ≥ 1
        Flat, even-length 1-D array of coordinate pairs.
        Layout: [x₀, y₀, x₁, y₁, …, xₙ₋₁, yₙ₋₁]
    lam : float
        Transformation intensity parameter λ.
        λ = 0  → identity (w = v)
        λ > 0  → counter-clockwise rotation + expansion
        λ < 0  → clockwise rotation + compression

    Returns
    -------
    np.ndarray, shape (2n,)
        Transformed vector w.
        w[0:2] ≡ v[0:2]  (anchor preserved exactly).

    Raises
    ------
    ValueError
        If len(v) < 2 or len(v) is odd.
    """
    if len(v) < 2:
        raise ValueError(f"Vector must have at least 2 elements; got {len(v)}.")
    if len(v) % 2 != 0:
        raise ValueError(f"Vector length must be even; got {len(v)}.")

    w       = v.copy()
    n_pairs = len(v) // 2

    # --- Derive transform parameters from the anchor pair ---
    ax, ay = float(v[0]), float(v[1])
    r0     = math.sqrt(ax * ax + ay * ay)           # Euclidean norm of anchor

    theta  = lam * math.atan2(ay, ax)               # Rotation angle (radians)
    s      = math.exp(lam * math.tanh(r0))          # Bounded scale factor

    cos_t  = math.cos(theta)
    sin_t  = math.sin(theta)

    # --- Transform each non-anchor coordinate pair ---
    for k in range(1, n_pairs):
        i      = 2 * k
        x, y   = float(v[i]), float(v[i + 1])

        # 2-D rotation-scaling:   [w_x]   =   s · R(θ) · [x]
        #                         [w_y]                   [y]
        w[i]     = s * (x * cos_t - y * sin_t)     # w_x
        w[i + 1] = s * (x * sin_t + y * cos_t)     # w_y

    return w


# ============================================================
# SECTION 5: DATASET GENERATION — RAW (UNIFORM SAMPLING)
# ============================================================

def generate_dataset_raw(
    v: np.ndarray,
    n_samples: int = 400,
) -> np.ndarray:
    """
    Generate a clean dataset using UNIFORM λ sampling (no softmax weighting).

    PURPOSE
    -------
    This function provides a baseline / ablation-study dataset where
    every λ candidate is equally likely.  It deliberately ignores the
    _ALPHA log-weights, so the empirical λ frequencies should converge
    to 1/|Λ| for each candidate as n_samples → ∞.

    Use this dataset to:
    - Isolate the effect of the softmax weighting (compare against
      generate_dataset_weighted).
    - Test decoders under a flat prior over λ.
    - Generate balanced class distributions for supervised learning.

    SAMPLING STRATEGY
    -----------------
    Each sample draws λ independently and uniformly:

        λ ~ Uniform({λ₁, λ₂, …, λₙ})   →   P(λᵢ) = 1/n  ∀ i

    Numpy's `np.random.choice` without a `p` argument performs this
    uniform draw efficiently.

    NOISE
    -----
    None.  All samples are noise-free and exactly invertible.

    Parameters
    ----------
    v : np.ndarray, shape (2m,),  m ≥ 1
        Ground-truth input vector (flat, even-length coordinate pairs).
        The vector is not modified; copies are transformed per sample.
    n_samples : int, optional
        Number of output samples.  Default: 400.
        Increase for better empirical frequency alignment with 1/|Λ|.

    Returns
    -------
    np.ndarray, shape (n_samples, len(v))
        Each row is transform(v, λᵢ) for a uniformly drawn λᵢ.
        Rows are independent; row order reflects draw order.

    Notes
    -----
    With the default _LAMBDA_SET of 5 candidates, the expected count
    per λ is n_samples / 5 = 80 for n_samples = 400.

    Example
    -------
    >>> v = np.array([1.0, 2.0, 3.0, 4.0])
    >>> ds = generate_dataset_raw(v, n_samples=10)
    >>> ds.shape
    (10, 4)
    """
    dataset = []

    for _ in range(n_samples):
        # Uniform draw: no probability weights passed to np.random.choice.
        lam = np.random.choice(_LAMBDA_SET)
        w   = transform(v, lam)
        dataset.append(w)

    return np.array(dataset)


# ============================================================
# SECTION 6: DATASET GENERATION — WEIGHTED (SOFTMAX SAMPLING, NO NOISE)
# ============================================================

def generate_dataset_weighted(
    v: np.ndarray,
    n_samples: int = 400,
) -> np.ndarray:
    """
    Generate a clean dataset using SOFTMAX-WEIGHTED λ sampling.

    PURPOSE
    -------
    This is the primary clean dataset.  The softmax distribution
    derived from _ALPHA controls how often each λ appears, allowing
    designers to bias the dataset toward specific transform intensities
    without ad-hoc frequency manipulation.

    Comparing this dataset against generate_dataset_raw reveals the
    direct effect of the softmax weighting on sample distribution.

    SAMPLING STRATEGY
    -----------------
    Each sample draws λ from the pre-computed softmax distribution:

        λ ~ Categorical(_LAMBDA_SET, _LAMBDA_PROBS)

        where  _LAMBDA_PROBS = softmax(_ALPHA)

    With the default _ALPHA = [−1, 0, 1, 0, −1]:
        λ =  0.00  →  ~58.3 % of samples  (identity transform)
        λ = ±0.05  →  ~21.5 % each
        λ = ±0.10  →   ~7.9 % each

    Samples drawn with λ = 0 are identical to v (no transformation).
    With default settings, roughly 58 % of the dataset is unmodified.

    NOISE
    -----
    None.  All samples are noise-free and exactly invertible:

        transform(transform(v, λ), −λ) = v   ∀ v, λ

    Parameters
    ----------
    v : np.ndarray, shape (2m,),  m ≥ 1
        Ground-truth input vector (flat, even-length coordinate pairs).
    n_samples : int, optional
        Number of output samples.  Default: 400.
        Larger values give better empirical agreement with
        the theoretical softmax probabilities.

    Returns
    -------
    np.ndarray, shape (n_samples, len(v))
        Each row is transform(v, λᵢ) for a softmax-drawn λᵢ.

    Notes
    -----
    To recover the true λ for a given row w, apply:
        transform(w, −λ)  for each candidate λ in _LAMBDA_SET
    and check which inverse reconstruction matches v via np.allclose.

    Example
    -------
    >>> v = np.array([1.0, 2.0, 3.0, 4.0])
    >>> ds = generate_dataset_weighted(v, n_samples=10)
    >>> ds.shape
    (10, 4)
    """
    dataset = []

    for _ in range(n_samples):
        # Softmax-weighted draw; _LAMBDA_PROBS sums to 1 by construction.
        lam = np.random.choice(_LAMBDA_SET, p=_LAMBDA_PROBS)
        w   = transform(v, lam)
        dataset.append(w)

    return np.array(dataset)


# ============================================================
# SECTION 7: DATASET GENERATION — NOISY (SOFTMAX + GAUSSIAN NOISE)
# ============================================================

def generate_dataset_noisy(
    v: np.ndarray,
    n_samples: int = 400,
    sigma: float   = _SIGMA,
) -> np.ndarray:
    """
    Generate a noisy dataset using softmax-weighted λ sampling plus
    bounded multiplicative Gaussian white noise on non-anchor elements.

    PURPOSE
    -------
    Simulates realistic measurement / sensor noise while preserving:
    1. The anchor pair exactly (required for geometric decoding).
    2. Signal polarity (noise never flips the sign of a coordinate).
    3. Approximate magnitude (bounded within ±5 % of original).
    4. Statistical independence across elements (white noise).

    This dataset is intended for training and evaluating noise-robust
    decoders.  Comparing it against generate_dataset_weighted isolates
    the effect of the noise augmentation.

    PIPELINE PER SAMPLE
    -------------------
    Step 1 — λ sampling:
        λ ~ Categorical(_LAMBDA_SET, _LAMBDA_PROBS)   (softmax distribution)

    Step 2 — Clean transform:
        w = transform(v, λ)

    Step 3 — Noise injection (non-anchor elements only):
        For each index i ≥ 2:
            w_noisy[i] = sign(w[i]) · clamp(|w[i]| · N(1, σ²),
                                             0.95|w[i]|,
                                             1.05|w[i]|)
        Anchor elements w[0], w[1] are copied unchanged.

    WHY ONLY NON-ANCHOR ELEMENTS?
    ------------------------------
    The anchor pair (w[0], w[1]) is the geometric reference from which
    downstream decoders compute θ and s.  Perturbing it would corrupt
    the decoding frame and introduce systematic bias into ALL recovered
    coordinates — qualitatively worse than the small, bounded noise on
    individual non-anchor elements.  Keeping the anchor exact is
    therefore a deliberate architectural choice.

    NOISE PROPERTIES
    ----------------
        E[w_noisy[i]]  ≈ w[i]         (approximately unbiased; exact
                                        bias from clamping is negligible
                                        for σ ≪ 0.05)
        |w_noisy[i] − w[i]| ≤ 0.05|w[i]|   (hard ±5 % bound)
        Cov[w_noisy[i], w_noisy[j]]   = 0   for i ≠ j  (white / i.i.d.)

    INVERTIBILITY
    -------------
    Unlike datasets (1) and (2), noisy samples are NOT exactly
    invertible.  Applying transform(w_noisy, −λ) recovers an
    approximation v̂ of v with:

        |v̂[i] − v[i]| ≤ 0.05 · |v[i]|   ∀ i ≥ 2
        v̂[0:2] = v[0:2]   (anchor always exact)

    Parameters
    ----------
    v : np.ndarray, shape (2m,),  m ≥ 1
        Ground-truth input vector (flat, even-length coordinate pairs).
    n_samples : int, optional
        Number of noisy output samples.  Default: 400.
    sigma : float, optional
        Standard deviation of the Gaussian multiplier N(1, σ²).
        Default: _SIGMA (module constant = 0.02).
        · σ → 0    : noise vanishes; dataset approaches weighted dataset.
        · σ = 0.02 : subtle, realistic noise (±2 % typical deviation).
        · σ → 0.05 : clamp begins to dominate the noise distribution.
        · σ ≫ 0.05 : output approaches Uniform(0.95|x|, 1.05|x|).

    Returns
    -------
    np.ndarray, shape (n_samples, len(v))
        Each row is transform(v, λᵢ) + bounded Gaussian noise on [2:].
        Row layout:
            row[0:2]   — anchor (exact, identical to clean transform)
            row[2:]    — noisy non-anchor elements

    Notes
    -----
    Downstream decoding strategy (multi-sample):
        1. Read the anchor from any row (always exact).
        2. For each sample wᵢ and each candidate λⱼ, compute
               v_candidate = transform(wᵢ, −λⱼ).
        3. Score λⱼ by intra-cluster variance across samples.
        4. The λ with the lowest variance is the true λ.
        5. Average reconstructed v across the matching cluster.

    Example
    -------
    >>> v = np.array([1.0, 2.0, 3.0, 4.0])
    >>> ds = generate_dataset_noisy(v, n_samples=10, sigma=0.02)
    >>> ds.shape
    (10, 4)
    >>> # Anchor is preserved exactly in every row:
    >>> assert np.all(ds[:, 0:2] == v[0:2])
    """
    dataset = []

    for _ in range(n_samples):
        # Step 1: softmax-weighted λ draw.
        lam = np.random.choice(_LAMBDA_SET, p=_LAMBDA_PROBS)

        # Step 2: apply clean geometric transform.
        w = transform(v, lam)

        # Step 3: inject i.i.d. Gaussian noise to non-anchor elements.
        # The anchor (indices 0–1) is deliberately left unperturbed;
        # see docstring section "WHY ONLY NON-ANCHOR ELEMENTS?" above.
        w_noisy = w.copy()
        if len(w) > 2:                              # guard: only if non-anchor elements exist
            w_noisy[2:] = _apply_noise_to_vector(w[2:], sigma)

        dataset.append(w_noisy)

    return np.array(dataset)


# ============================================================
# SECTION 8: VERIFICATION HELPERS
# ============================================================

def verify_clean_invertibility(v: np.ndarray) -> None:
    """
    Verify that transform(transform(v, λ), −λ) = v for all λ in _LAMBDA_SET.

    Prints a formatted table showing the max element-wise reconstruction
    error for each λ and marks each row as PASS (✓) or FAIL (✗) using
    np.allclose with default tolerances (atol=1e-8, rtol=1e-5).

    This check is meaningful only for clean (noise-free) transforms.
    Noisy samples are not exactly invertible; see verify_noisy_approximation.

    Parameters
    ----------
    v : np.ndarray
        Ground-truth input vector.

    Returns
    -------
    None  (output is printed to stdout).
    """
    print("\n--- Invertibility Check (Clean Transform) ---")
    print(f"{'λ':>8}  {'Max Reconstruction Error':>26}  {'Status':>6}")
    print("-" * 50)

    for lam in _LAMBDA_SET:
        w     = transform(v, lam)
        v_rec = transform(w, -lam)
        err   = np.max(np.abs(v - v_rec))
        ok    = np.allclose(v, v_rec)
        print(f"{lam:>+8.2f}  {err:>26.2e}  {'✓' if ok else '✗':>6}")


def verify_noisy_approximation(v: np.ndarray, sigma: float = _SIGMA) -> None:
    """
    Demonstrate approximate reconstruction from a single noisy sample.

    For each λ in _LAMBDA_SET, generates one noisy sample and applies
    the inverse transform.  Reports mean and max element-wise error
    to illustrate the bounded-noise recovery property.

    Because a single sample is used, results are stochastic and will
    vary between runs.  The key expected observation is that max error
    stays at or below 5 % of the corresponding v element magnitude.

    Parameters
    ----------
    v : np.ndarray
        Ground-truth input vector.
    sigma : float, optional
        Noise standard deviation.  Default: _SIGMA.

    Returns
    -------
    None  (output is printed to stdout).
    """
    print("\n--- Approximate Recovery Check (Noisy Transform, single sample) ---")
    print(f"{'λ':>8}  {'Mean Error':>14}  {'Max Error':>12}")
    print("-" * 40)

    for lam in _LAMBDA_SET:
        w_clean       = transform(v, lam)
        w_noisy       = w_clean.copy()
        if len(w_noisy) > 2:
            w_noisy[2:] = _apply_noise_to_vector(w_clean[2:], sigma)

        v_approx = transform(w_noisy, -lam)
        errs     = np.abs(v - v_approx)

        print(f"{lam:>+8.2f}  {errs.mean():>14.4e}  {errs.max():>12.4e}")


def report_lambda_frequencies(
    dataset: np.ndarray,
    v: np.ndarray,
    label: str,
    atol: float = 1e-6,
) -> None:
    """
    Report empirical λ sampling frequencies for a given dataset.

    For each λ candidate, counts how many rows in `dataset` are
    consistent with that λ by applying the inverse transform and
    comparing against v.

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, len(v))
        Generated dataset (clean or noisy).
    v : np.ndarray
        Ground-truth vector used to generate the dataset.
    label : str
        Human-readable name for the dataset (used in printed header).
    atol : float, optional
        Absolute tolerance for np.allclose reconstruction check.
        Default 1e-6 suits clean datasets.  For noisy datasets use
        a relaxed value such as 0.1 to account for noise error.

    Returns
    -------
    None  (output is printed to stdout).
    """
    n = len(dataset)
    print(f"\nEmpirical λ frequencies — {label} (n={n}, atol={atol}):")
    print(f"{'λ':>8}  {'Count':>7}  {'Empirical%':>12}  {'Expected%':>12}")
    print("-" * 48)

    for lam, prob in zip(_LAMBDA_SET, _LAMBDA_PROBS):
        reconstructed = np.array([transform(w, -lam) for w in dataset])
        matches       = int(np.sum([np.allclose(r, v, atol=atol) for r in reconstructed]))
        print(
            f"{lam:>+8.2f}  {matches:>7d}  {100*matches/n:>11.1f}%"
            f"  {100*prob:>11.1f}%"
        )


# ============================================================
# SECTION 9: ENTRY POINT — USER INPUT
# ============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # 9a. Prompt the user for a ground-truth vector.
    # ----------------------------------------------------------
    print("=" * 60)
    print("  Nonlinear Causal Rotation-Scaling Transform  v3.0.0")
    print("=" * 60)

    while True:
        try:
            raw = input("\nEnter ground-truth vector (e.g. [1,2,3,4]): ")
            v   = np.array(eval(raw), dtype=float)  # noqa: S307 — controlled CLI input

            if len(v) < 2:
                raise ValueError("Vector must have at least 2 elements.")
            if len(v) % 2 != 0:
                raise ValueError("Vector length must be even (pairs of x, y).")

            break

        except Exception as exc:
            print(f"  Invalid input: {exc}. Please try again.")

    # ----------------------------------------------------------
    # 9b. Generate all three datasets.
    # ----------------------------------------------------------
    print(f"\n[1/3] Generating RAW dataset      (400 samples, uniform λ)  ...")
    dataset_raw = generate_dataset_raw(v, n_samples=400)

    print(f"[2/3] Generating WEIGHTED dataset  (400 samples, softmax λ)  ...")
    dataset_weighted = generate_dataset_weighted(v, n_samples=400)

    print(f"[3/3] Generating NOISY dataset     (400 samples, softmax λ, σ={_SIGMA}) ...")
    dataset_noisy = generate_dataset_noisy(v, n_samples=400, sigma=_SIGMA)

    # ----------------------------------------------------------
    # 9c. Verification checks.
    # ----------------------------------------------------------
    verify_clean_invertibility(v)
    verify_noisy_approximation(v)

    # ----------------------------------------------------------
    # 9d. Empirical λ frequency reports for each dataset.
    # ----------------------------------------------------------
    # Raw dataset: expected uniform ~ 20 % each, but we compare
    # against softmax probs for reference; the mismatch is intentional.
    report_lambda_frequencies(dataset_raw,      v, label="RAW      (uniform sample)",   atol=1e-6)
    report_lambda_frequencies(dataset_weighted, v, label="WEIGHTED (softmax sample)",   atol=1e-6)
    report_lambda_frequencies(dataset_noisy,    v, label="NOISY    (softmax + noise)",  atol=0.10)

    # ----------------------------------------------------------
    # 9e. Save all six files (3 datasets × 2 formats each).
    # ----------------------------------------------------------
    os.makedirs(_SAVE_DIR, exist_ok=True)

    _paths = {
        "raw":      ("dataset_raw.csv",      "dataset_raw.npy",      dataset_raw),
        "weighted": ("dataset_weighted.csv", "dataset_weighted.npy", dataset_weighted),
        "noisy":    ("dataset_noisy.csv",    "dataset_noisy.npy",    dataset_noisy),
    }

    print(f"\n--- Saving datasets to: {_SAVE_DIR} ---")
    for key, (csv_name, npy_name, ds) in _paths.items():
        csv_path = os.path.join(_SAVE_DIR, csv_name)
        npy_path = os.path.join(_SAVE_DIR, npy_name)
        np.savetxt(csv_path, ds, delimiter=",")
        np.save(npy_path, ds)
        print(f"  [{key:>8}]  CSV → {csv_path}")
        print(f"  [{key:>8}]  NPY → {npy_path}")

    # ----------------------------------------------------------
    # 9f. Display sample output from each dataset.
    # ----------------------------------------------------------
    for label, ds in [("RAW", dataset_raw), ("WEIGHTED", dataset_weighted), ("NOISY", dataset_noisy)]:
        print(f"\nFirst 5 {label} samples (rounded to 3 d.p.):")
        print(np.round(ds[:5], 3))

    # Element-wise noise delta between first weighted and noisy samples.
    if len(v) > 2:
        delta = np.abs(dataset_weighted[0, 2:] - dataset_noisy[0, 2:])
        print(f"\nNoise delta on sample-0 non-anchor elements "
              f"(WEIGHTED vs NOISY):  max={delta.max():.4f}  mean={delta.mean():.4f}")