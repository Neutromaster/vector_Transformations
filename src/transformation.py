import numpy as np
import os

# ============================================================
# MODULE: Nonlinear Causal Rotation-Scaling Transform
# ============================================================

"""
OVERVIEW
--------
This module implements a nonlinear, causal transformation on vectors
interpreted as sequences of 2D coordinate pairs.

Each transformation uses a hidden parameter λ (lambda), randomly chosen
from a fixed discrete set. The parameter is NOT exposed to the user.

Multiple transformed outputs generated from the same input vector can be
used to recover the original vector via consistency analysis.

------------------------------------------------------------
CORE IDEA
------------------------------------------------------------
Input vector:
    v = [x0, y0, x1, y1, x2, y2, ...]

Output:
    w = T(v, λ)

Transformation rules:
- First pair remains unchanged
- Each subsequent pair is rotated and scaled based on the anchor pair

------------------------------------------------------------
MATHEMATICAL FORMULATION
------------------------------------------------------------
Let anchor = (x0, y0)

    r0 = ||anchor||
    θ  = sign(λ) * atan2(y0, x0)
    s  = exp(λ * r0)

For k ≥ 1:

    w_k = s · R(θ) · v_k

------------------------------------------------------------
KEY PROPERTIES
------------------------------------------------------------
- Nonlinear (state-dependent scaling via anchor)
- Globally consistent transform (single θ and s)
- Hidden parameter λ
- Single output not invertible (λ unknown)
- Multiple outputs → recoverable system

------------------------------------------------------------
INTENDED USE
------------------------------------------------------------
- Inverse problems
- SIMC-style modelling challenges
- Parameter inference systems
"""



# ============================================================
# INTERNAL CONFIGURATION (HIDDEN PARAMETER SET)
# ============================================================

# Discrete set of λ values sampled during dataset generation.
# These are intentionally hidden from the user — the inference
# challenge is to recover v without knowing which λ was used.
_LAMBDA_SET = [0.1, -0.05, 0, 0.05, 0.1]


# ============================================================
# TRANSFORM FUNCTION
# ============================================================

def transform(v, lam):
    """
    Apply the anchor-based nonlinear rotation-scaling transform.

    The first coordinate pair acts as a fixed anchor that determines
    a single rotation angle and scaling factor applied uniformly to
    all subsequent pairs. This makes the transform globally consistent
    and exactly invertible via transform(w, -lam).

    Parameters
    ----------
    v : numpy.ndarray of shape (2n,)
        Input vector of n 2D coordinate pairs.
    lam : float
        Transformation parameter λ. Pass -λ to invert.

    Returns
    -------
    w : numpy.ndarray
        Transformed vector of same shape as v.
    """

    # Output vector — first pair is always carried over unchanged,
    # as it serves as the anchor for all downstream computations
    w = v.copy()

    # Total number of 2D coordinate pairs in the input vector
    n_pairs = len(v) // 2

    # ----------------------------------------------------------
    # ANCHOR EXTRACTION
    # Extract the first pair (ax, ay) — this is the fixed reference
    # point that drives the single global rotation and scale
    # ----------------------------------------------------------
    ax, ay = v[0], v[1]

    # Euclidean magnitude of the anchor pair
    # Controls the strength of the exponential scaling
    r0 = np.sqrt(ax**2 + ay**2)

    # Rotation angle derived from the anchor's direction
    # sign(λ) ensures the rotation reverses when λ is negated,
    # which is the key property that makes the transform invertible
    theta = np.sign(lam) * np.arctan2(ay, ax)

    # Exponential scaling factor — grows/shrinks all pairs uniformly
    # Using exp ensures s(-λ) = 1/s(λ), giving exact invertibility
    s = np.exp(lam * r0)

    # Precompute rotation matrix components for efficiency
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # ----------------------------------------------------------
    # PAIR-WISE TRANSFORMATION
    # Apply the same R(θ) and s to every non-anchor pair.
    # ----------------------------------------------------------
    for k in range(1, n_pairs):

        # Extract the k-th input coordinate pair
        x, y = v[2*k], v[2*k + 1]

        # Apply rotation followed by scaling:
        #   w_k = s · R(θ) · v_k
        w[2*k]     = s * (x * cos_t - y * sin_t)
        w[2*k + 1] = s * (x * sin_t + y * cos_t)

    return w


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(v, n_samples=400):
    """
    Generate multiple transformed outputs from the same input vector.

    Each sample uses a different (hidden) λ.

    Parameters
    ----------
    v : array-like
        Ground truth vector
    n_samples : int
        Number of transformed outputs

    Returns
    -------
    dataset : numpy array (n_samples x len(v))
        Collection of transformed vectors
    """

    # Accumulator for transformed output vectors
    dataset = []

    for _ in range(n_samples):

        # Sample a λ uniformly at random from the hidden discrete set.
        lam = np.random.choice(_LAMBDA_SET)

        # Apply the transform and store the result
        w = transform(v, lam)
        dataset.append(w)

    # Stack into a 2D array
    return np.array(dataset)


# ============================================================
# USER INPUT (HIDDEN GROUND TRUTH VIA eval)
# ============================================================

# Ground truth vector is provided by user.
# IMPORTANT:
# - eval() is used as explicitly requested
# - The vector is NEVER printed or exposed anywhere in output

while True:
    try:
        _user_input = input(
            "Enter ground truth vector (e.g. [x0, y0, x1, y1, ...]): "
        )

        # Convert user input into numpy array
        v = np.array(eval(_user_input), dtype=float)

        # Ensure valid structure (pairs of coordinates)
        if len(v) % 2 != 0:
            raise ValueError("Vector length must be even.")

        break

    except Exception:
        print("Invalid input. Please enter a valid list of numbers.")


# ============================================================
# DATASET GENERATION
# ============================================================

# Generate the full dataset of transformed outputs
dataset = generate_dataset(v, n_samples=400)


# ============================================================
# REVERSIBILITY CHECK
# ============================================================

# Verify that transform(transform(v, λ), -λ) == v
# WITHOUT printing or exposing v

print("\n--- Reversibility Check ---")
_all_passed = True

for lam in _LAMBDA_SET:

    w = transform(v, lam)
    v_recovered = transform(w, -lam)

    # Only check equality — do NOT print vectors
    passed = np.allclose(v, v_recovered, atol=1e-10)
    print(f"  λ = {lam:+.2f} | Match: {passed}")

    if not passed:
        _all_passed = False

print(f"\nAll λ values reversible: {_all_passed}")


# ============================================================
# SAVE DATASET TO LOCAL PATH
# ============================================================

_SAVE_DIR = r"C:\Users\DELL\Documents\GitHub\vector_Transformations\data"
os.makedirs(_SAVE_DIR, exist_ok=True)

_dataset_path      = os.path.join(_SAVE_DIR, "transformed_dataset.npy")

# Save both arrays (ground truth is stored but NOT printed)
np.save(_dataset_path, dataset)

print(f"\nDataset saved to:      {_dataset_path}")


# ============================================================
# LOAD DATASET FROM LOCAL PATH
# ============================================================

dataset_loaded      = np.load(_dataset_path)

print("\n--- Loaded Dataset ---")
print(f"Dataset shape:       {dataset_loaded.shape}")

print("\nFirst 5 loaded transformed outputs:")
print(np.round(dataset_loaded[:5], 3))


# ============================================================
# DISPLAY FULL DATASET AS LIST
# ============================================================

print("\n--- Full Transformed Dataset ---")
for i, sample in enumerate(dataset_loaded):
    print(f"  [{i:>3}]: {np.round(sample, 3).tolist()}")