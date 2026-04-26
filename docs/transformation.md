# `transformation.py` — Full Documentation

**Source:** `src/transformation.py`
**Module:** Controlled Nonlinear Causal Rotation–Scaling Transform
**Version:** 1.0.0

This document is written for someone who has been handed the generated
dataset and must **recover the hidden ground-truth vector `v`**. It
explains the transform, the softmax-normalised λ sampling, and the
properties that make inversion tractable.

---

## 1. Overview

The module implements a nonlinear, anchor-conditioned, causal
transformation on a vector

```
v = [x0, y0, x1, y1, x2, y2, ..., x_{n-1}, y_{n-1}]
```

interpreted as `n` 2-D coordinate pairs `p_0, p_1, ..., p_{n-1}` where
`p_k = (x_k, y_k)`.

A hidden scalar parameter **λ**, drawn from a fixed discrete set
`Λ = _LAMBDA_SET` according to a **softmax-normalised** probability
distribution over designer-specified log-weights, controls the
transform. Each call to `generate_dataset` produces many outputs
`w = T(v, λ_i)` — one for each sampled λ. The solver's job is to
recover `v` from the collection of `w`s without knowing which λ
produced each one.

The design simultaneously achieves:

- **Deterministic, reproducible output ratios** (controlled by `α`)
- **Smooth probabilistic weighting** with no arbitrary frequency bias
- **Exact invertibility** — `T(T(v, λ), −λ) ≡ v`
- **Anchor causality** — the first coordinate pair governs the
  rotation angle and scale applied to all subsequent pairs

---

## 2. The Transform, Step by Step

### 2.1 The anchor

The first pair `p_0 = (x0, y0)` is the **anchor**. It has two roles:

1. It is copied into the output unchanged: `w_0 = p_0`.
2. It alone determines the rotation angle and scale factor applied
   to every other pair.

> **Key consequence:** every output vector `w` begins with the exact
> same `(x0, y0)` as the input `v`. The first pair of the hidden
> vector is therefore **not hidden at all** — it is visible in every
> sample of the dataset.

### 2.2 Anchor-derived quantities

From the anchor the transform computes three scalars:

```
r0   = sqrt(x0² + y0²)            # anchor magnitude
θ(λ) = λ · atan2(y0, x0)          # rotation angle (anchor-conditioned)
s(λ) = exp(λ · tanh(r0))          # bounded scale factor
```

Properties of these quantities:

| Quantity | Depends on | Notes                                                                  |
|----------|------------|------------------------------------------------------------------------|
| `r0`     | `v` only   | Constant across λ.                                                     |
| `θ(λ)`   | `v`, λ     | Linear in λ. `θ(0) = 0`. `θ(−λ) = −θ(λ)`.                              |
| `s(λ)`   | `v`, λ     | Strictly positive. `s(0) = 1`. `s(−λ) = 1/s(λ)`. **Bounded** via `tanh`. |

The `tanh(r0)` factor is what bounds the scale: because
`tanh : [0, ∞) → [0, 1)`, the exponent `λ · tanh(r0)` cannot diverge
for finite λ even if `r0` is large. This keeps `s(λ)` numerically
well-behaved for any input magnitude.

### 2.3 Per-pair update

For every `k ≥ 1`:

```
w_k = s(λ) · R(θ(λ)) · p_k
```

where `R(θ)` is the standard 2-D rotation matrix

```
R(θ) = [ cos θ   -sin θ ]
       [ sin θ    cos θ ]
```

Written out in components:

```
w_k.x = s · (x_k · cos θ  −  y_k · sin θ)
w_k.y = s · (x_k · sin θ  +  y_k · cos θ)
```

**The same `s` and `θ` are applied to every non-anchor pair inside a
single output `w`.** The transform is globally consistent within a
sample — it is not pair-specific.

---

## 3. λ Sampling — Softmax Normalisation

λ is not chosen uniformly. It is sampled from a fixed discrete set
according to a **softmax-normalised** probability distribution
parameterised by designer-supplied **log-weights** `α`.

### 3.1 Definitions

```
Λ = _LAMBDA_SET = [-0.10, -0.05, 0.00, 0.05, 0.10]   # 5 candidates
α = _ALPHA      = [-1.0,  0.0,  1.0,  0.0, -1.0]     # log-weights
P(λ_i)          = softmax(α_i) = exp(α_i) / Σ_j exp(α_j)
```

### 3.2 Why softmax?

Softmax converts an unconstrained real-valued log-weight vector `α`
into a valid probability distribution that automatically satisfies:

- `P(λ_i) > 0` for all i,
- `Σ_i P(λ_i) = 1`,
- ratios `P(λ_i) / P(λ_j) = exp(α_i − α_j)` depend only on **differences**
  of log-weights — adding a constant to every `α_i` leaves the
  distribution unchanged.

The designer therefore specifies the *shape* of the distribution by
choosing the relative magnitudes of the `α_i`s, not the absolute
probabilities. To make λ_i twice as likely as λ_j, set
`α_i = α_j + ln 2`. To make a value very rare, push its `α` strongly
negative.

### 3.3 Numerical stability

```python
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)
```

Subtracting `max(x)` before exponentiation is mathematically a no-op
(the constant cancels in numerator and denominator) but ensures the
largest exponent is `exp(0) = 1`, preventing overflow in `exp` for
large `α` values.

### 3.4 Resulting distribution under the default α

With `α = [−1, 0, +1, 0, −1]` the softmax produces:

| λ        | α    | P(λ) (≈) |
|----------|------|----------|
| −0.10    | −1.0 |  ~7.9 %  |
| −0.05    |  0.0 | ~21.5 %  |
| **0.00** | +1.0 | **~58.3 %** |
| +0.05    |  0.0 | ~21.5 %  |
| +0.10    | −1.0 |  ~7.9 %  |

The distribution is **symmetric about zero** and **peaked at λ = 0**.
Roughly 58% of generated samples are *identity* outputs (i.e. exact
copies of `v`); the remaining ~42% spread between mild and strong
forward/inverse rotation–scaling.

### 3.5 Implementation

```python
_LAMBDA_PROBS = softmax(_ALPHA)             # computed once at import
lam = np.random.choice(_LAMBDA_SET, p=_LAMBDA_PROBS)
```

`_LAMBDA_PROBS` is precomputed at module load time for efficiency.

---

## 4. Worked Micro-Example

Take `v = [3, 4, 1, 0, 0, 2]` and `λ = 0.1`.

```
r0       = sqrt(3² + 4²)        = 5
tanh(r0) = tanh(5)              ≈ 0.99991
θ        = 0.1 · atan2(4, 3)    ≈ +0.09273 rad
s        = exp(0.1 · 0.99991)   ≈ 1.10516
```

Anchor copied: `w_0 = (3, 4)`.

Pair `p_1 = (1, 0)`:

```
w_1.x = 1.10516 · (1·cos 0.09273 − 0·sin 0.09273) ≈ 1.10041
w_1.y = 1.10516 · (1·sin 0.09273 + 0·cos 0.09273) ≈ 0.10240
```

Pair `p_2 = (0, 2)`:

```
w_2.x = 1.10516 · (0·cos 0.09273 − 2·sin 0.09273) ≈ −0.20479
w_2.y = 1.10516 · (0·sin 0.09273 + 2·cos 0.09273) ≈  2.20081
```

Output: `w ≈ [3, 4, 1.100, 0.102, −0.205, 2.201]`.

---

## 5. Key Properties (Useful for Inversion)

1. **Anchor preservation.**
   `w[0:2] == v[0:2]` for every sample and every λ.

2. **Exact invertibility given λ.**
   Because `s(−λ) = 1/s(λ)` and `θ(−λ) = −θ(λ)`, applying the
   transform again with `−λ` undoes it:
   `transform(transform(v, λ), −λ) == v`
   exactly (up to floating-point noise — the script verifies this in
   Section 7 of the source with `np.allclose`).

3. **λ = 0 is the identity.**
   If λ = 0 then `s = 1` and `θ = 0`, so `w == v`. With the default
   softmax-normalised distribution, ~58% of samples will be exact
   copies of `v`.

4. **Finite hidden set.**
   `_LAMBDA_SET` contains five distinct values. Every sample in the
   dataset therefore comes from one of five `(s, θ)` pairs that are
   **fully determined by the anchor** — there is no continuous
   parameter to estimate.

5. **Uniform geometry inside a sample.**
   For any two non-anchor pairs `p_j, p_k` inside the **same**
   output `w`:
   `||w_j|| / ||w_k|| == ||p_j|| / ||p_k||`
   and the angle between `w_j` and `w_k` equals the angle between
   `p_j` and `p_k`. The transform rotates and rescales them
   **together**, so their relative geometry is preserved.

6. **Cross-sample magnitude ratios reveal `s`.**
   For two samples `w^{(a)}` and `w^{(b)}` of the same pair index
   `k ≥ 1`:
   `||w_k^{(a)}|| / ||w_k^{(b)}|| == s(λ_a) / s(λ_b)`,
   independent of `p_k`.

7. **Empirical frequencies converge to softmax probabilities.**
   For a dataset of size `N`, the count of samples produced by each
   λ_i approaches `N · P(λ_i)` as `N → ∞`. Cluster sizes in the
   recovered λ-labelling are therefore a direct empirical estimate of
   the softmax distribution.

---

## 6. Recovery Strategy

Given a dataset of `N` samples `{w^{(i)}}`, a clean path to `v` is:

### Step A — Read the anchor directly.

`(x0, y0) = w^{(i)}[0:2]` for **any** `i`. Compute
`r0 = sqrt(x0² + y0²)`.

### Step B — Enumerate the five candidate `(s, θ)` pairs.

Since `_LAMBDA_SET` has five distinct values and `r0, x0, y0` are
known, you can compute all five candidate `(s(λ), θ(λ))` pairs
explicitly using the formulae in §2.2 — even though λ itself is
"hidden", the finite set of possible transforms it induces is fully
determined by the anchor.

### Step C — Classify each sample.

For each sample `w^{(i)}` and each candidate λ_j, compute

```
v_candidate(i, j) = transform(w^{(i)}, −λ_j)
```

Score each candidate λ_j by **intra-cluster variance** of the
recovered `v_candidate(·, j)` across all samples. Low variance →
candidate λ is consistent → likely the true λ for the samples that
agree under it. The correct λ for sample `i` is the one whose
inverse-applied output coincides with the inverse-applied outputs of
many other samples.

### Step D — Invert.

Once each sample is labelled with its λ, apply `transform(w, −λ)` to
recover `v` from any sample. With multiple samples in the same λ
cluster you can average to suppress floating-point noise.

### Shortcut.

Any sample whose non-anchor pairs are **identical** to those of the
anchor (within tolerance) was generated with λ = 0 and **equals `v`
directly**. With the default `α` configuration, ~58% of samples fall
into this category, so the shortcut is almost guaranteed to succeed
on a dataset of any reasonable size.

---

## 7. API Reference

### `softmax(x)`

Numerically stable softmax over a 1-D array of real-valued log-weights.

| Parameter | Type            | Description                       |
|-----------|-----------------|-----------------------------------|
| `x`       | `numpy.ndarray` | 1-D log-weight vector.            |

Returns a probability vector of the same shape: entries are positive
and sum to 1. Uses the max-subtraction trick for stability.

---

### `transform(v, lam)`

Apply the anchor-based nonlinear rotation–scaling transform.

| Parameter | Type                             | Description                                               |
|-----------|----------------------------------|-----------------------------------------------------------|
| `v`       | `numpy.ndarray` of shape `(2n,)` | Input vector of `n` 2-D coordinate pairs. `n ≥ 1`.        |
| `lam`     | `float`                          | Transformation parameter λ. Pass `−λ` to invert.          |

Returns `w : numpy.ndarray` of the same shape as `v`.

Behaviour:

- `w[0:2] = v[0:2]` (anchor copied unchanged).
- For `k ≥ 1`, `w[2k:2k+2] = s · R(θ) · v[2k:2k+2]`,
  with `s = exp(λ · tanh(r0))` and `θ = λ · atan2(y0, x0)`.

Raises `ValueError` if `v` has odd length or fewer than 2 elements.

---

### `generate_dataset(v, n_samples=400)`

Generate multiple transformed outputs from the same input vector.
Each sample independently draws λ from `_LAMBDA_SET` according to the
**softmax-normalised** distribution `_LAMBDA_PROBS`.

| Parameter   | Type        | Description                                  |
|-------------|-------------|----------------------------------------------|
| `v`         | array-like  | Ground truth vector (flat, even-length).     |
| `n_samples` | `int`       | Number of transformed outputs. Default 400.  |

Returns a `(n_samples, len(v))` NumPy array. The λ value used for
each row is **not** stored — recovering it is part of the challenge.

Larger `n_samples` improves the empirical frequency alignment with
the theoretical softmax distribution.

---

## 8. Module Configuration

| Constant         | Value                              | Meaning                                |
|------------------|------------------------------------|----------------------------------------|
| `_LAMBDA_SET`    | `[−0.10, −0.05, 0.00, 0.05, 0.10]` | Discrete λ candidates.                 |
| `_ALPHA`         | `[−1.0, 0.0, 1.0, 0.0, −1.0]`      | Log-weights for the softmax.           |
| `_LAMBDA_PROBS`  | `softmax(_ALPHA)`                  | Sampling probabilities (auto-derived). |
| `_SAVE_DIR`      | `…/vector_Transformations/data`    | Output directory for dataset files.    |

To change the sampling ratios, edit `_ALPHA`. To change the available
intensities, edit `_LAMBDA_SET`. The two arrays must have equal length.

---

## 9. Module-Level Script Behaviour

Running `python src/transformation.py` executes:

1. **Prompt** the user for a ground-truth vector, parsed with `eval`.
   The vector must be non-empty with even length; it is never echoed.
2. **Generate** 400 transformed outputs via `generate_dataset`,
   sampling λ from the softmax-normalised distribution.
3. **Reversibility check** — for each λ in `_LAMBDA_SET`, confirm that
   `transform(transform(v, λ), −λ) == v` using `np.allclose`. Prints
   the maximum reconstruction error and a ✓/✗ flag per λ.
4. **Save** the dataset to:
   - `…/data/dataset.csv` (CSV via `np.savetxt`)
   - `…/data/dataset.npy` (binary via `np.save`)
5. **Print** the first five rows rounded to three decimals.
6. **Empirical frequency report** — for each λ, count how many rows
   reconstruct exactly to `v` under `transform(w, −λ)` and compare
   the empirical fraction to the theoretical `_LAMBDA_PROBS[i]`.

The ground-truth vector `v` and the per-sample λ are never printed.

---

## 10. Dependencies

- `numpy`
- `os`, `csv` (standard library)

---

## 11. Summary Cheat-Sheet

| Quantity        | Formula                              | Known to solver?                |
|-----------------|--------------------------------------|---------------------------------|
| `(x0, y0)`      | `w[0:2]` of any sample               | ✅ directly                     |
| `r0`            | `sqrt(x0² + y0²)`                    | ✅ derived                      |
| `θ(λ)`          | `λ · atan2(y0, x0)`                  | ✅ for each λ in `_LAMBDA_SET`  |
| `s(λ)`          | `exp(λ · tanh(r0))`                  | ✅ for each λ in `_LAMBDA_SET`  |
| `P(λ)`          | `softmax(α)`                         | ✅ if `α` is published          |
| `λ` per sample  | not stored                           | ❌ must be inferred             |
| `p_k, k ≥ 1`    | `(1/s) · R(−θ) · w_k`                | recoverable once λ is known    |
