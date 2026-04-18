import numpy as np
import os
import csv

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

    w = v.copy()
    n_pairs = len(v) // 2

    ax, ay = v[0], v[1]
    r0 = np.sqrt(ax**2 + ay**2)

    theta = np.sign(lam) * np.arctan2(ay, ax)
    s = np.exp(lam * r0)

    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for k in range(1, n_pairs):
        x, y = v[2*k], v[2*k + 1]

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

    dataset = []

    for _ in range(n_samples):
        lam = np.random.choice(_LAMBDA_SET)
        w = transform(v, lam)
        dataset.append(w)

    return np.array(dataset)

# ============================================================
# USER INPUT (HIDDEN GROUND TRUTH VIA eval)
# ============================================================

while True:
    try:
        _user_input = input(
            "Enter ground truth vector (e.g. [x0, y0, x1, y1, ...]): "
        )

        v = np.array(eval(_user_input), dtype=float)

        if len(v) % 2 != 0:
            raise ValueError("Vector length must be even.")

        break

    except Exception:
        print("Invalid input. Please enter a valid list of numbers.")

# ============================================================
# DATASET GENERATION
# ============================================================

dataset = generate_dataset(v, n_samples=400)

# ============================================================
# REVERSIBILITY CHECK
# ============================================================

print("\n--- Reversibility Check ---")
_all_passed = True

for lam in _LAMBDA_SET:
    w = transform(v, lam)
    v_recovered = transform(w, -lam)

    passed = np.allclose(v, v_recovered, atol=1e-10)
    print(f"  λ = {lam:+.2f} | Match: {passed}")

    if not passed:
        _all_passed = False

print(f"\nAll λ values reversible: {_all_passed}")

# ============================================================
# SAVE DATASET TO LOCAL PATH (CSV + NPY)
# ============================================================

_SAVE_DIR = r"C:\Users\DELL\Documents\GitHub\vector_Transformations\data"
os.makedirs(_SAVE_DIR, exist_ok=True)

_csv_path = os.path.join(_SAVE_DIR, "transformed_dataset.csv")
_npy_path = os.path.join(_SAVE_DIR, "transformed_dataset.npy")

# ---- Save CSV ----
headers = [f"x{i//2}" if i % 2 == 0 else f"y{i//2}" for i in range(len(v))]

with open(_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(dataset)

print(f"\nDataset saved to CSV: {_csv_path}")

# ---- Save NPY ----
np.save(_npy_path, dataset)

print(f"Dataset saved to NPY: {_npy_path}")

# ============================================================
# LOAD DATASET FROM LOCAL PATH
# ============================================================

# ---- Load CSV ----
dataset_csv = []

with open(_csv_path, mode='r') as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
        dataset_csv.append([float(x) for x in row])

dataset_csv = np.array(dataset_csv)

# ---- Load NPY ----
dataset_npy = np.load(_npy_path)

print("\n--- Loaded Dataset (CSV) ---")
print(f"Dataset shape: {dataset_csv.shape}")

print("\n--- Loaded Dataset (NPY) ---")
print(f"Dataset shape: {dataset_npy.shape}")

print("\nFirst 5 loaded transformed outputs (CSV):")
print(np.round(dataset_csv[:5], 3))

print("\nFirst 5 loaded transformed outputs (NPY):")
print(np.round(dataset_npy[:5], 3))

# ============================================================
# CONSISTENCY CHECK
# ============================================================

print("\n--- Consistency Check (CSV vs NPY) ---")
same = np.allclose(dataset_csv, dataset_npy)
print(f"Datasets match: {same}")

# ============================================================
# DISPLAY FULL DATASET AS LIST
# ============================================================

print("\n--- Full Transformed Dataset ---")
for i, sample in enumerate(dataset_npy):
    print(f"  [{i:>3}]: {np.round(sample, 3).tolist()}")