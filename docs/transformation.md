# `transformation.py` — Full Documentation

**Source:** `src/transformation.py`
**Module:** Controlled Nonlinear Causal Rotation–Scaling Transform
            with Optional Gaussian White Noise Augmentation
**Version:** 2.0.0

This document is written for someone who has been handed the generated
datasets and must **recover the hidden ground-truth vector `v`**. It
explains the transform, the softmax-normalised λ sampling, the new
Gaussian white noise augmentation, and the properties that make
inversion (exact or approximate) tractable.

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
- **Exact invertibility** (clean transform) — `T(T(v, λ), −λ) ≡ v`
- **Anchor causality** — the first coordinate pair governs the
  rotation angle and scale applied to all subsequent pairs

### 1.1 Version 2.0 additions

Version 2.0 adds **Gaussian white noise augmentation** via a bounded
multiplicative noise model:

```
x_noisy = x · N(1, σ²),  clamped to [0.95·|x|, 1.05·|x|]
```

This simulates realistic sensor/measurement noise while preserving:

- **Signal polarity** — the sign of each element is never flipped.
- **Approximate magnitude** — bounded within ±5 % of the original.
- **Statistical structure** — noise is i.i.d. across elements (white).

The script now produces **four** datasets in total:

| # | File                  | Format | Contents                          |
|---|-----------------------|--------|-----------------------------------|
| 1 | `dataset_clean.csv`   | CSV    | Original transform, no noise      |
| 2 | `dataset_clean.npy`   | NumPy  | Original transform, no noise      |
| 3 | `dataset_noisy.csv`   | CSV    | Transform + Gaussian white noise  |
| 4 | `dataset_noisy.npy`   | NumPy  | Transform + Gaussian white noise  |

Invertibility holds **exactly** for clean samples and only
**approximately** for noisy samples (see §6 and §8).

---

## 2. The Transform, Step by Step

### 2.1 The anchor

The first pair `p_0 = (x0, y0)` is the **anchor**. It has two roles:

1. It is copied into the output unchanged: `w_0 = p_0`.
2. It alone determines the rotation angle and scale factor applied
   to every other pair.

> **Key consequence:** every clean output vector `w` begins with the
> exact same `(x0, y0)` as the input `v`. The first pair of the hidden
> vector is therefore **not hidden at all** — it is visible in every
> sample. The anchor is **also kept noise-free in the noisy dataset**
> (see §4.4) so this property holds across all four output files.

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
Roughly 58 % of generated samples are *identity* outputs (i.e. exact
copies of `v` in the clean dataset, or noisy copies of `v` in the
noisy dataset); the remaining ~42 % spread between mild and strong
forward/inverse rotation–scaling.

### 3.5 Implementation

```python
_LAMBDA_PROBS = softmax(_ALPHA)             # computed once at import
lam = np.random.choice(_LAMBDA_SET, p=_LAMBDA_PROBS)
```

`_LAMBDA_PROBS` is precomputed at module load time for efficiency
and is reused identically by both `generate_dataset` and
`generate_dataset_noisy`.

---

## 4. Gaussian White Noise Augmentation (v2.0)

### 4.1 Noise model

For each non-anchor element `x` of a clean transformed sample:

```
m       ~ N(1, σ²)                      # multiplicative noise
output  = sign(x) · clip( |x| · m,  0.95·|x|,  1.05·|x| )
```

The noise is:

- **Multiplicative**, not additive — perturbation magnitude scales
  with signal magnitude, keeping the signal-to-noise ratio (SNR)
  roughly constant across the dynamic range of the data.
- **White (i.i.d.)** — every element draws an independent multiplier;
  no two elements share a draw, so there is zero correlation between
  components. This is the defining property of white noise (vs.
  coloured noise, where adjacent elements are correlated).
- **Unbiased in expectation** — `E[m] = 1`, so `E[x_noisy] = x`.
- **Bounded** — magnitude is hard-clamped to ±5 % of `|x|`,
  independent of σ. This truncates the Gaussian tails and matches
  the behaviour of real bounded sensors.
- **Polarity-preserving** — `sign(x_noisy) = sign(x)` is enforced
  explicitly. Noise can never flip the sign of a coordinate.

### 4.2 Properties of the σ–clamp interaction

The ±5 % hard clamp is **independent of σ**:

| σ regime            | Effect of the clamp                                   |
|---------------------|-------------------------------------------------------|
| `σ << 0.05`         | Most draws are inside the clamp; clamp rarely fires.  |
| `σ ≈ 0.05`          | Mixed regime; some Gaussian shape, some clipping.     |
| `σ >> 0.05`         | Clamp dominates; distribution ≈ uniform on `[0.95|x|, 1.05|x|]`. |

The default `_SIGMA = 0.02` puts ~95 % of multipliers in `[0.96, 1.04]`
**before** clamping, so clipping is rare and the perturbation is
effectively a soft Gaussian wobble of a few percent.

### 4.3 Edge case: zero

If an element is exactly `0.0`, multiplicative noise has no
well-defined scale (`0 · m = 0` for any `m`), so the function returns
`0.0` immediately. No randomness is consumed for zero inputs.

### 4.4 Where noise is applied (and where it isn't)

`apply_noise_to_vector` applies noise to **every** element of the
input vector it receives — the anchor included. However,
`generate_dataset_noisy` deliberately **slices off the anchor**
before calling it:

```python
w_noisy = w.copy()
w_noisy[2:] = apply_noise_to_vector(w[2:], sigma)
```

so the anchor pair `w[0:2]` is preserved exactly, matching the
clean transform's anchor-preservation guarantee. This keeps the
anchor usable as the geometric reference for rotation/scale
recovery during decoding.

> Whether to noise the anchor or not is a design choice. The library
> exposes both: call `apply_noise_to_vector` directly to noise
> everything, or call `generate_dataset_noisy` to noise only
> non-anchor elements.

---

## 5. Worked Micro-Example (Clean)

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

Clean output: `w ≈ [3, 4, 1.100, 0.102, −0.205, 2.201]`.

Applying noise with `σ = 0.02` then yields, for example,

```
w_noisy ≈ [3, 4, 1.108, 0.101, −0.207, 2.183]
```

— same anchor, every other element nudged within ±5 %.

---

## 6. Key Properties (Useful for Inversion)

1. **Anchor preservation.**
   `w[0:2] == v[0:2]` for every sample and every λ, in **both** the
   clean and noisy datasets.

2. **Exact invertibility — clean only.**
   Because `s(−λ) = 1/s(λ)` and `θ(−λ) = −θ(λ)`, applying the
   transform again with `−λ` undoes it:
   `transform(transform(v, λ), −λ) == v`
   exactly (up to floating-point noise — the script verifies this in
   §9 of the source with `np.allclose`).

3. **Approximate invertibility — noisy.**
   For noisy samples,
   `transform(w_noisy, −λ) ≈ v`
   with per-element error bounded by the ±5 % clamp width.
   The script reports the mean and max reconstruction error per λ
   in §10 of the source.

4. **λ = 0 is the identity (in the clean transform).**
   If λ = 0 then `s = 1` and `θ = 0`, so `w == v`. With the default
   softmax-normalised distribution, ~58 % of clean samples are exact
   copies of `v`. In the noisy dataset, λ = 0 samples are
   element-wise noisy copies of `v` (anchor still exact).

5. **Finite hidden set.**
   `_LAMBDA_SET` contains five distinct values. Every sample in the
   dataset therefore comes from one of five `(s, θ)` pairs that are
   **fully determined by the anchor** — there is no continuous
   parameter to estimate.

6. **Uniform geometry inside a clean sample.**
   For any two non-anchor pairs `p_j, p_k` inside the **same** clean
   output `w`:
   `||w_j|| / ||w_k|| == ||p_j|| / ||p_k||`
   and the angle between `w_j` and `w_k` equals the angle between
   `p_j` and `p_k`. Noise breaks this exact equality but only by a
   bounded multiplicative factor close to 1.

7. **Cross-sample magnitude ratios reveal `s`.**
   For two clean samples `w^{(a)}` and `w^{(b)}` of the same pair
   index `k ≥ 1`:
   `||w_k^{(a)}|| / ||w_k^{(b)}|| == s(λ_a) / s(λ_b)`,
   independent of `p_k`. In the noisy dataset, the ratio holds in
   expectation but with a small i.i.d. multiplicative jitter.

8. **Empirical frequencies converge to softmax probabilities.**
   For a dataset of size `N`, the count of samples produced by each
   λ_i approaches `N · P(λ_i)` as `N → ∞`. Cluster sizes in the
   recovered λ-labelling are therefore a direct empirical estimate of
   the softmax distribution. For the noisy dataset, frequency
   estimation requires a tolerance-based comparison (the script uses
   `atol=0.1` in §13).

---

## 7. Recovery Strategy

Given a clean dataset of `N` samples `{w^{(i)}}`, a clean path to `v`
is described in 7.1. Noisy datasets follow the same strategy with the
adjustments described in 7.2.

### 7.1 Clean dataset

#### Step A — Read the anchor directly.

`(x0, y0) = w^{(i)}[0:2]` for **any** `i`. Compute
`r0 = sqrt(x0² + y0²)`.

#### Step B — Enumerate the five candidate `(s, θ)` pairs.

Since `_LAMBDA_SET` has five distinct values and `r0, x0, y0` are
known, you can compute all five candidate `(s(λ), θ(λ))` pairs
explicitly using the formulae in §2.2 — even though λ itself is
"hidden", the finite set of possible transforms it induces is fully
determined by the anchor.

#### Step C — Classify each sample.

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

#### Step D — Invert.

Once each sample is labelled with its λ, apply `transform(w, −λ)` to
recover `v` from any sample. With multiple samples in the same λ
cluster you can average to suppress floating-point noise.

#### Shortcut.

Any sample whose non-anchor pairs are **identical** to those of the
anchor (within tolerance) was generated with λ = 0 and **equals `v`
directly**. With the default `α` configuration, ~58 % of clean
samples fall into this category.

### 7.2 Noisy dataset

The same four-step strategy still works, with two modifications:

1. **Use a tolerance.** Replace exact equality with
   `np.allclose(..., atol=ε)` where ε is roughly the clamp width
   (5 % of typical element magnitude). The script uses `atol=0.1` for
   its sanity check.
2. **Average within clusters.** Because each noisy sample carries
   independent perturbations, averaging the inverse-applied outputs
   inside the same λ-cluster reduces residual error proportionally to
   `1/√N_cluster`. This is the main reason the noisy dataset still
   permits accurate recovery: many samples → noise averages out;
   the structural transform does not.

The exact-equality shortcut is **not available** for the noisy
dataset — even λ = 0 samples are perturbed.

---

## 8. API Reference

### `softmax(x)`

Numerically stable softmax over a 1-D array of real-valued log-weights.

| Parameter | Type            | Description                       |
|-----------|-----------------|-----------------------------------|
| `x`       | `numpy.ndarray` | 1-D log-weight vector.            |

Returns a probability vector of the same shape: entries are positive
and sum to 1. Uses the max-subtraction trick for stability.

---

### `apply_noise_to_element(x, sigma)`

Apply bounded multiplicative Gaussian white noise to a single scalar.

| Parameter | Type    | Description                                                 |
|-----------|---------|-------------------------------------------------------------|
| `x`       | `float` | Original scalar. If `x == 0`, returns `0.0` immediately.    |
| `sigma`   | `float` | Std-dev of the multiplier `N(1, σ²)`. Typical: 0.01 – 0.10. |

Returns a `float` with the same sign as `x` and magnitude in
`[0.95·|x|, 1.05·|x|]`.

Behaviour:

- Draws `m ~ N(1, σ²)`.
- Computes `|x| · m`, clamps it to `[0.95·|x|, 1.05·|x|]`.
- Restores `sign(x)` to the result.

The ±5 % hard clamp is **independent of σ**.

---

### `apply_noise_to_vector(vec, sigma)`

Apply independent Gaussian white noise to every element of a vector.

| Parameter | Type            | Description                                       |
|-----------|-----------------|---------------------------------------------------|
| `vec`     | `numpy.ndarray` | Flat 1-D array. Shape preserved in the output.    |
| `sigma`   | `float`         | Passed through to `apply_noise_to_element`.       |

Returns a new `numpy.ndarray` of the same shape as `vec`.

Each element is processed independently — the noise is **i.i.d.**
across the vector. The function does not exclude the anchor; if you
want anchor-preservation, slice it off before calling
(see `generate_dataset_noisy`).

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

Generate multiple **clean** transformed outputs from the same input
vector. Each sample independently draws λ from `_LAMBDA_SET` according
to the **softmax-normalised** distribution `_LAMBDA_PROBS`.

| Parameter   | Type        | Description                                  |
|-------------|-------------|----------------------------------------------|
| `v`         | array-like  | Ground truth vector (flat, even-length).     |
| `n_samples` | `int`       | Number of transformed outputs. Default 400.  |

Returns a `(n_samples, len(v))` NumPy array. The λ value used for
each row is **not** stored — recovering it is part of the challenge.
Samples are exactly invertible.

---

### `generate_dataset_noisy(v, n_samples=400, sigma=_SIGMA)`

Generate multiple **noisy** transformed outputs. Each sample is
produced by:

1. Drawing λ from the softmax-normalised distribution.
2. Applying the clean geometric transform.
3. Injecting i.i.d. multiplicative Gaussian white noise into the
   non-anchor elements (anchor is preserved exactly).

| Parameter   | Type        | Description                                                         |
|-------------|-------------|---------------------------------------------------------------------|
| `v`         | array-like  | Ground truth vector (flat, even-length).                            |
| `n_samples` | `int`       | Number of noisy outputs. Default 400.                               |
| `sigma`     | `float`     | Std-dev of the multiplicative noise. Default `_SIGMA` (= 0.02).     |

Returns a `(n_samples, len(v))` NumPy array. Anchor elements `[:, 0:2]`
match the clean transform exactly; all other elements carry bounded
multiplicative noise. Noisy samples are **not exactly invertible**
but reconstruct `v` to within the clamp width per element.

---

## 9. Module Configuration

| Constant         | Value                              | Meaning                                       |
|------------------|------------------------------------|-----------------------------------------------|
| `_LAMBDA_SET`    | `[−0.10, −0.05, 0.00, 0.05, 0.10]` | Discrete λ candidates.                        |
| `_ALPHA`         | `[−1.0, 0.0, 1.0, 0.0, −1.0]`      | Log-weights for the softmax.                  |
| `_LAMBDA_PROBS`  | `softmax(_ALPHA)`                  | Sampling probabilities (auto-derived).        |
| `_SIGMA`         | `0.02`                             | Default σ for Gaussian noise multiplier.      |
| `_SAVE_DIR`      | `…/vector_Transformations/data`    | Output directory for dataset files.           |

To change the sampling ratios, edit `_ALPHA`. To change the available
intensities, edit `_LAMBDA_SET`. The two arrays must have equal length.
To change the noise strength globally, edit `_SIGMA`; per-call
overrides are also supported via the `sigma` parameter of
`generate_dataset_noisy`.

---

## 10. Module-Level Script Behaviour

Running `python src/transformation.py` executes:

1. **Prompt** the user for a ground-truth vector, parsed with `eval`.
   The vector must be non-empty with even length; it is never echoed.
2. **Generate the clean dataset** — 400 transformed outputs via
   `generate_dataset`, sampling λ from the softmax-normalised
   distribution.
3. **Generate the noisy dataset** — 400 transformed + noised outputs
   via `generate_dataset_noisy(σ=_SIGMA)`. Anchor is preserved.
4. **Reversibility check (clean only)** — for each λ in `_LAMBDA_SET`,
   confirm that `transform(transform(v, λ), −λ) == v` using
   `np.allclose`. Prints the maximum reconstruction error and a ✓/✗
   flag per λ.
5. **Approximate reconstruction check (noisy)** — for each λ,
   generate a single noisy sample, invert with `−λ`, and report the
   mean and max element-wise error. Errors are bounded by the ±5 %
   clamp.
6. **Save four datasets** to:
   - `…/data/dataset_clean.csv` (CSV via `np.savetxt`)
   - `…/data/dataset_clean.npy` (binary via `np.save`)
   - `…/data/dataset_noisy.csv` (CSV via `np.savetxt`)
   - `…/data/dataset_noisy.npy` (binary via `np.save`)
7. **Print** the first five rows of each dataset rounded to three
   decimals, plus a quick noise-magnitude summary
   (max / mean |clean − noisy|) for sample 0.
8. **Empirical frequency report** — for each λ, count how many rows
   reconstruct to `v` under `transform(w, −λ)` and compare the
   empirical fraction to the theoretical `_LAMBDA_PROBS[i]`.
   Reported separately for the clean dataset (exact match,
   `np.allclose`) and the noisy dataset (tolerance match,
   `np.allclose(..., atol=0.1)`).

The ground-truth vector `v` and the per-sample λ are never printed.

---

## 11. Dependencies

- `numpy`
- `os`, `csv`, `math`, `random` (Python standard library)

`random.gauss` is used internally by `apply_noise_to_element` for the
multiplicative noise draw; `numpy.random.choice` is used for
softmax-weighted λ sampling.

---

## 12. Summary Cheat-Sheet

| Quantity            | Formula                              | Known to solver?                      |
|---------------------|--------------------------------------|---------------------------------------|
| `(x0, y0)`          | `w[0:2]` of any sample (any dataset) | ✅ directly                           |
| `r0`                | `sqrt(x0² + y0²)`                    | ✅ derived                            |
| `θ(λ)`              | `λ · atan2(y0, x0)`                  | ✅ for each λ in `_LAMBDA_SET`        |
| `s(λ)`              | `exp(λ · tanh(r0))`                  | ✅ for each λ in `_LAMBDA_SET`        |
| `P(λ)`              | `softmax(α)`                         | ✅ if `α` is published                |
| `λ` per sample      | not stored                           | ❌ must be inferred                   |
| `p_k, k ≥ 1` (clean)| `(1/s) · R(−θ) · w_k`                | exact once λ is known                 |
| `p_k, k ≥ 1` (noisy)| `(1/s) · R(−θ) · w_k`                | approximate; ±5 % per element bound;  |
|                     |                                      | average over a λ-cluster to denoise   |
| Noise model         | `x · N(1, σ²)`, clipped to ±5 %·|x|  | ✅ if `σ` is published (default 0.02) |
