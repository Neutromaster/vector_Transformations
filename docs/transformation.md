# `transformation.py` — Full Documentation

**Source:** `src/transformation.py`
**Module:** Controlled Nonlinear Causal Rotation–Scaling Transform
            with Optional Softmax-Weighted Sampling and
            Optional Gaussian White Noise Augmentation
**Version:** 4.0.0

This document is written for someone who has been handed the generated
datasets and must **recover the hidden ground-truth vector `v`**. It
explains the transform, the optional softmax-normalised λ sampling,
the optional Gaussian white noise augmentation, and the properties
that make inversion (exact or approximate) tractable.

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
`Λ = _LAMBDA_SET`, controls the transform. The library exposes
**four** dataset generators that together form a 2 × 2 design matrix
crossing two orthogonal choices:

|                          | **No noise**            | **+ Bounded Gaussian**          |
|--------------------------|-------------------------|---------------------------------|
| **Uniform λ sampling**   | `generate_dataset_raw`  | `generate_dataset_noisy`        |
| **Softmax λ sampling**   | `generate_dataset_weighted` | `generate_dataset_weighted_noisy` |

The two orthogonal axes are:

- **Sampling distribution** — uniform over `Λ` vs softmax-weighted by
  `α = _ALPHA`.
- **Noise** — clean transform output vs bounded multiplicative
  Gaussian noise on non-anchor elements.

The solver's job is to recover `v` from the collection of `w`s without
knowing which λ produced each one.

The design simultaneously achieves:

- **Deterministic, reproducible output ratios** (controlled by `α`)
- **Smooth probabilistic weighting** with no arbitrary frequency bias
- **Exact invertibility** (clean transforms) — `T(T(v, λ), −λ) ≡ v`
- **Anchor causality** — the first coordinate pair governs the
  rotation angle and scale applied to all subsequent pairs
- **Anchor preservation under noise** — even in the noisy datasets,
  `w[0:2] == v[0:2]` exactly

### 1.1 Output files

The script produces **eight** files in total — 4 datasets × 2 formats
(CSV via `np.savetxt` and binary NumPy via `np.save`):

| # | File                          | Format | λ sampling | Noise | Contents                                   |
|---|-------------------------------|--------|------------|-------|--------------------------------------------|
| 1 | `dataset_raw.csv`             | CSV    | Uniform    |  No   | Clean transform, flat λ prior              |
| 2 | `dataset_raw.npy`             | NumPy  | Uniform    |  No   | Clean transform, flat λ prior              |
| 3 | `dataset_weighted.csv`        | CSV    | Softmax    |  No   | Clean transform, designer-weighted λ       |
| 4 | `dataset_weighted.npy`        | NumPy  | Softmax    |  No   | Clean transform, designer-weighted λ       |
| 5 | `dataset_noisy.csv`           | CSV    | Uniform    | Yes   | Transform + Gaussian white noise           |
| 6 | `dataset_noisy.npy`           | NumPy  | Uniform    | Yes   | Transform + Gaussian white noise           |
| 7 | `dataset_weighted_noisy.csv`  | CSV    | Softmax    | Yes   | Transform + Gaussian white noise           |
| 8 | `dataset_weighted_noisy.npy`  | NumPy  | Softmax    | Yes   | Transform + Gaussian white noise           |

Invertibility holds **exactly** for clean datasets (1–4) and only
**approximately** for noisy datasets (5–8); see §6 and §8.

### 1.2 Why four datasets?

The 2 × 2 design lets you cleanly separate the contributions of the
sampling distribution from the contributions of the noise:

- **Effect of softmax weighting on a clean signal** — compare
  `dataset_raw` against `dataset_weighted`.
- **Effect of softmax weighting on a noisy signal** — compare
  `dataset_noisy` against `dataset_weighted_noisy`.
- **Effect of noise under a flat λ prior** — compare `dataset_raw`
  against `dataset_noisy`.
- **Effect of noise under a softmax λ prior** — compare
  `dataset_weighted` against `dataset_weighted_noisy`.

Each pairwise comparison holds one axis fixed and varies the other,
which is the canonical setup for ablation studies and noise-robust
decoder evaluation.

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
> sample. The anchor is **also kept noise-free in the noisy datasets**
> (see §4.4) so this property holds across all eight output files.

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

## 3. λ Sampling — Uniform vs Softmax

The four datasets pair the same `_LAMBDA_SET` with two different
sampling distributions. The distribution is the only thing that
differs between `raw` ↔ `weighted` and between `noisy` ↔
`weighted_noisy`.

### 3.1 Definitions

```
Λ = _LAMBDA_SET = [-0.10, -0.05, 0.00, 0.05, 0.10]   # 5 candidates
α = _ALPHA      = [-1.0,  0.0,  1.0,  0.0, -1.0]     # log-weights

# Uniform (used by raw, noisy):
P_uniform(λ_i) = 1 / |Λ| = 0.20

# Softmax (used by weighted, weighted_noisy):
P_softmax(λ_i) = exp(α_i) / Σ_j exp(α_j)
```

### 3.2 Why offer both?

**Uniform** sampling is a flat prior — every λ candidate is equally
likely. This is the natural baseline / ablation distribution: it
divorces the dataset's empirical λ frequencies from the designer's
log-weights, which is useful for benchmarking decoders under no prior
information about λ.

**Softmax** sampling lets the designer bias the dataset toward
specific transform intensities by editing `α` — without ad-hoc
frequency manipulation. Softmax converts an unconstrained
real-valued log-weight vector `α` into a valid probability
distribution that automatically satisfies:

- `P(λ_i) > 0` for all i,
- `Σ_i P(λ_i) = 1`,
- ratios `P(λ_i) / P(λ_j) = exp(α_i − α_j)` depend only on **differences**
  of log-weights — adding a constant to every `α_i` leaves the
  distribution unchanged.

To make `λ_i` twice as likely as `λ_j`, set `α_i = α_j + ln 2`. To
make a value very rare, push its `α` strongly negative.

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

### 3.4 Resulting distributions under the default α

Uniform (`raw`, `noisy`):

| λ     | P(λ) |
|-------|------|
| −0.10 | 20 % |
| −0.05 | 20 % |
|  0.00 | 20 % |
| +0.05 | 20 % |
| +0.10 | 20 % |

Softmax (`weighted`, `weighted_noisy`) with `α = [−1, 0, +1, 0, −1]`:

| λ        | α    | P(λ) (≈) |
|----------|------|----------|
| −0.10    | −1.0 |  ~7.9 %  |
| −0.05    |  0.0 | ~21.5 %  |
| **0.00** | +1.0 | **~58.3 %** |
| +0.05    |  0.0 | ~21.5 %  |
| +0.10    | −1.0 |  ~7.9 %  |

The softmax distribution is **symmetric about zero** and **peaked at
λ = 0**. Roughly 58 % of softmax-sampled clean rows are *identity*
outputs (i.e. exact copies of `v` in the `weighted` dataset, or noisy
copies of `v` in the `weighted_noisy` dataset). The remaining ~42 %
spread between mild and strong forward/inverse rotation–scaling.

In the uniform datasets, only ~20 % of rows correspond to λ = 0.

### 3.5 Implementation

```python
_LAMBDA_PROBS = softmax(_ALPHA)             # computed once at import

# Uniform (raw, noisy):
lam = np.random.choice(_LAMBDA_SET)

# Softmax (weighted, weighted_noisy):
lam = np.random.choice(_LAMBDA_SET, p=_LAMBDA_PROBS)
```

`_LAMBDA_PROBS` is precomputed at module load time for efficiency
and is reused identically by `generate_dataset_weighted` and
`generate_dataset_weighted_noisy`.

---

## 4. Gaussian White Noise Augmentation

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

The same noise model is shared by `generate_dataset_noisy` and
`generate_dataset_weighted_noisy`. The only difference between the
two functions is the λ sampling distribution.

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

`_apply_noise_to_vector` applies noise to **every** element of the
input vector it receives — the anchor included. However, both
`generate_dataset_noisy` and `generate_dataset_weighted_noisy`
deliberately **slice off the anchor** before calling it:

```python
w_noisy = w.copy()
w_noisy[2:] = _apply_noise_to_vector(w[2:], sigma)
```

so the anchor pair `w[0:2]` is preserved exactly, matching the
clean transform's anchor-preservation guarantee. This keeps the
anchor usable as the geometric reference for rotation/scale
recovery during decoding.

> Whether to noise the anchor or not is a design choice. The library
> exposes both: call `_apply_noise_to_vector` directly to noise
> everything, or call `generate_dataset_noisy` /
> `generate_dataset_weighted_noisy` to noise only non-anchor elements.

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

This single per-sample pipeline is identical for `dataset_noisy` and
`dataset_weighted_noisy`; the two datasets differ only in *how* `λ`
is drawn before this pipeline runs.

---

## 6. Key Properties (Useful for Inversion)

1. **Anchor preservation.**
   `w[0:2] == v[0:2]` for every sample and every λ, in **all four**
   datasets.

2. **Exact invertibility — clean datasets only (`raw`, `weighted`).**
   Because `s(−λ) = 1/s(λ)` and `θ(−λ) = −θ(λ)`, applying the
   transform again with `−λ` undoes it:
   `transform(transform(v, λ), −λ) == v`
   exactly (up to floating-point noise — the script verifies this in
   §9 of the source with `np.allclose`).

3. **Approximate invertibility — noisy datasets (`noisy`, `weighted_noisy`).**
   For noisy samples,
   `transform(w_noisy, −λ) ≈ v`
   with per-element error bounded by the ±5 % clamp width.
   The script reports the mean and max reconstruction error per λ
   in §10 of the source.

4. **λ = 0 is the identity (for the clean transform stage).**
   If λ = 0 then `s = 1` and `θ = 0`, so the clean transform output
   is exactly `v`.
   - In `raw` and `weighted`, this means λ = 0 samples are byte-for-byte
     copies of `v`.
   - In `noisy` and `weighted_noisy`, λ = 0 samples are noisy copies
     of `v` (anchor still exact, non-anchor elements within ±5 %).

   The expected fraction of λ = 0 samples is **20 %** in the uniform
   datasets and **~58 %** in the softmax datasets.

5. **Finite hidden set.**
   `_LAMBDA_SET` contains five distinct values. Every sample in any
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
   independent of `p_k`. In the noisy datasets, the ratio holds in
   expectation but with a small i.i.d. multiplicative jitter.

8. **Empirical frequencies converge to the chosen prior.**
   For a dataset of size `N`, the count of samples produced by each
   λ_i approaches:
   - `N / |Λ|`           in `raw` and `noisy`            (uniform), and
   - `N · P_softmax(λ_i)` in `weighted` and `weighted_noisy` (softmax).

   Cluster sizes in the recovered λ-labelling are therefore a direct
   empirical estimate of whichever prior was used. For the noisy
   datasets, frequency estimation requires a tolerance-based
   comparison (the script uses `atol=0.1` in §13).

---

## 7. Recovery Strategy

Given a dataset of `N` samples `{w^{(i)}}`, a path to `v` is described
in 7.1 for the clean datasets. Noisy datasets follow the same strategy
with the adjustments described in 7.2.

### 7.1 Clean datasets (`raw`, `weighted`)

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
anchor's geometry under λ = 0 (within tolerance) was generated with
λ = 0 and **equals `v` directly**. Expected fractions:
- `raw`      — 20 % of samples (uniform prior).
- `weighted` — ~58 % of samples (softmax prior with default α).

### 7.2 Noisy datasets (`noisy`, `weighted_noisy`)

The same four-step strategy still works, with two modifications:

1. **Use a tolerance.** Replace exact equality with
   `np.allclose(..., atol=ε)` where ε is roughly the clamp width
   (5 % of typical element magnitude). The script uses `atol=0.1` for
   its sanity check.
2. **Average within clusters.** Because each noisy sample carries
   independent perturbations, averaging the inverse-applied outputs
   inside the same λ-cluster reduces residual error proportionally to
   `1/√N_cluster`. This is the main reason the noisy datasets still
   permit accurate recovery: many samples → noise averages out;
   the structural transform does not.

The exact-equality shortcut is **not available** for the noisy
datasets — even λ = 0 samples are perturbed.

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

### `_apply_noise_to_scalar(x, sigma)`

*Module-private.* Apply bounded multiplicative Gaussian white noise
to a single scalar.

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

### `_apply_noise_to_vector(vec, sigma)`

*Module-private.* Apply independent Gaussian white noise to every
element of a vector.

| Parameter | Type            | Description                                       |
|-----------|-----------------|---------------------------------------------------|
| `vec`     | `numpy.ndarray` | Flat 1-D array. Shape preserved in the output.    |
| `sigma`   | `float`         | Passed through to `_apply_noise_to_scalar`.       |

Returns a new `numpy.ndarray` of the same shape as `vec`.

Each element is processed independently — the noise is **i.i.d.**
across the vector. The function does not exclude the anchor; if you
want anchor-preservation, slice it off before calling (this is how
both noisy dataset generators use it internally).

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

### `generate_dataset_raw(v, n_samples=400)`

Generate a **clean** dataset using **uniform** λ sampling. Every
candidate in `_LAMBDA_SET` is drawn with equal probability `1/|Λ|`,
independently of `_ALPHA`.

| Parameter   | Type        | Description                                  |
|-------------|-------------|----------------------------------------------|
| `v`         | array-like  | Ground truth vector (flat, even-length).     |
| `n_samples` | `int`       | Number of transformed outputs. Default 400.  |

Returns a `(n_samples, len(v))` NumPy array. Samples are exactly
invertible. Expected count per λ is `n_samples / |Λ|` (= 80 for
`n_samples = 400` and `|Λ| = 5`).

---

### `generate_dataset_weighted(v, n_samples=400)`

Generate a **clean** dataset using **softmax-weighted** λ sampling.
Each sample independently draws λ from `_LAMBDA_SET` according to
the softmax-normalised distribution `_LAMBDA_PROBS = softmax(_ALPHA)`.

| Parameter   | Type        | Description                                  |
|-------------|-------------|----------------------------------------------|
| `v`         | array-like  | Ground truth vector (flat, even-length).     |
| `n_samples` | `int`       | Number of transformed outputs. Default 400.  |

Returns a `(n_samples, len(v))` NumPy array. The λ value used for
each row is **not** stored — recovering it is part of the challenge.
Samples are exactly invertible.

---

### `generate_dataset_noisy(v, n_samples=400, sigma=_SIGMA)`

Generate a **noisy** dataset using **uniform** λ sampling. Each
sample is produced by:

1. Drawing λ uniformly from `_LAMBDA_SET`.
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

This is the noise-augmented analogue of `generate_dataset_raw`.

---

### `generate_dataset_weighted_noisy(v, n_samples=400, sigma=_SIGMA)`

Generate a **noisy** dataset using **softmax-weighted** λ sampling.
Each sample is produced by:

1. Drawing λ from `_LAMBDA_SET` according to `_LAMBDA_PROBS = softmax(_ALPHA)`.
2. Applying the clean geometric transform.
3. Injecting i.i.d. multiplicative Gaussian white noise into the
   non-anchor elements (anchor is preserved exactly).

| Parameter   | Type        | Description                                                         |
|-------------|-------------|---------------------------------------------------------------------|
| `v`         | array-like  | Ground truth vector (flat, even-length).                            |
| `n_samples` | `int`       | Number of noisy outputs. Default 400.                               |
| `sigma`     | `float`     | Std-dev of the multiplicative noise. Default `_SIGMA` (= 0.02).     |

Returns a `(n_samples, len(v))` NumPy array with the same anchor /
noise properties as `generate_dataset_noisy`.

This is the noise-augmented analogue of `generate_dataset_weighted`
and the most "production-realistic" of the four datasets — it
combines designer-controlled λ weighting with sensor-style bounded
noise.

---

### `verify_clean_invertibility(v)`

Print a per-λ table reporting the max element-wise error of
`transform(transform(v, λ), −λ) − v`, with a ✓/✗ flag from
`np.allclose`. Useful as a sanity check that the clean transform is
implemented correctly.

---

### `verify_noisy_approximation(v, sigma=_SIGMA)`

Print a per-λ table reporting the mean and max element-wise error of
inverse-applying a single noisy sample. Demonstrates that errors stay
within the ±5 % clamp width. Shared between `noisy` and
`weighted_noisy` since both use the same noise model.

---

### `report_lambda_frequencies(dataset, v, label, atol=1e-6)`

For each candidate λ, count how many rows of `dataset` are consistent
with that λ via `transform(w, −λ) ≈ v`. Print empirical percentage
side-by-side with the softmax probability column.

> **Note.** The "Softmax%" column is shown for *all* datasets,
> including the uniform ones (`raw`, `noisy`), as a constant
> cross-dataset reference. For uniform datasets, the empirical
> frequencies should converge to `1/|Λ|` (= 20 % per λ for `|Λ| = 5`),
> not to the softmax column. Don't be alarmed by the apparent
> mismatch in the uniform reports — it is intentional.

---

## 9. Module Configuration

| Constant         | Value                              | Meaning                                       |
|------------------|------------------------------------|-----------------------------------------------|
| `_LAMBDA_SET`    | `[−0.10, −0.05, 0.00, 0.05, 0.10]` | Discrete λ candidates.                        |
| `_ALPHA`         | `[−1.0, 0.0, 1.0, 0.0, −1.0]`      | Log-weights for the softmax.                  |
| `_LAMBDA_PROBS`  | `softmax(_ALPHA)`                  | Sampling probabilities (auto-derived).        |
| `_SIGMA`         | `0.02`                             | Default σ for Gaussian noise multiplier.      |
| `_SAVE_DIR`      | `…/vector_Transformations/data`    | Output directory for dataset files.           |

To change the softmax sampling ratios, edit `_ALPHA`. To change the
available intensities, edit `_LAMBDA_SET`. The two arrays must have
equal length. To change the noise strength globally, edit `_SIGMA`;
per-call overrides are also supported via the `sigma` parameter of
`generate_dataset_noisy` and `generate_dataset_weighted_noisy`.

---

## 10. Module-Level Script Behaviour

Running `python src/transformation.py` executes:

1. **Prompt** the user for a ground-truth vector, parsed with `eval`.
   The vector must be non-empty with even length; it is never echoed.
2. **Generate the raw dataset** — 400 transformed outputs via
   `generate_dataset_raw`, sampling λ uniformly from `_LAMBDA_SET`.
3. **Generate the weighted dataset** — 400 transformed outputs via
   `generate_dataset_weighted`, sampling λ from the softmax-normalised
   distribution.
4. **Generate the noisy dataset** — 400 transformed + noised outputs
   via `generate_dataset_noisy(σ=_SIGMA)`. λ is sampled uniformly;
   anchor is preserved.
5. **Generate the weighted+noisy dataset** — 400 transformed + noised
   outputs via `generate_dataset_weighted_noisy(σ=_SIGMA)`. λ is
   softmax-sampled; anchor is preserved.
6. **Reversibility check (clean only)** — for each λ in `_LAMBDA_SET`,
   confirm that `transform(transform(v, λ), −λ) == v` using
   `np.allclose`. Prints the maximum reconstruction error and a ✓/✗
   flag per λ.
7. **Approximate reconstruction check (noisy)** — for each λ,
   generate a single noisy sample, invert with `−λ`, and report the
   mean and max element-wise error. Errors are bounded by the ±5 %
   clamp.
8. **Empirical frequency report** — for each of the four datasets,
   for each λ, count how many rows reconstruct to `v` under
   `transform(w, −λ)` and compare the empirical fraction to the
   softmax reference column. Reported with `atol=1e-6` for clean
   datasets and `atol=0.1` for noisy datasets.
9. **Save eight datasets** to:
   - `…/data/dataset_raw.csv`            (CSV via `np.savetxt`)
   - `…/data/dataset_raw.npy`            (binary via `np.save`)
   - `…/data/dataset_weighted.csv`       (CSV via `np.savetxt`)
   - `…/data/dataset_weighted.npy`       (binary via `np.save`)
   - `…/data/dataset_noisy.csv`          (CSV via `np.savetxt`)
   - `…/data/dataset_noisy.npy`          (binary via `np.save`)
   - `…/data/dataset_weighted_noisy.csv` (CSV via `np.savetxt`)
   - `…/data/dataset_weighted_noisy.npy` (binary via `np.save`)
10. **Print** the first five rows of each of the four datasets rounded
    to three decimals, plus a quick noise-magnitude summary
    (max / mean |weighted − weighted_noisy|) for sample 0. The
    weighted vs weighted+noisy comparison holds the λ-sampling
    distribution fixed and isolates the noise effect.

The ground-truth vector `v` and the per-sample λ are never printed.

---

## 11. Dependencies

- `numpy`
- `os`, `math`, `random` (Python standard library)

`random.gauss` is used internally by `_apply_noise_to_scalar` for the
multiplicative noise draw; `numpy.random.choice` is used for both
uniform and softmax-weighted λ sampling.

---

## 12. Summary Cheat-Sheet

| Quantity                  | Formula                              | Known to solver?                      |
|---------------------------|--------------------------------------|---------------------------------------|
| `(x0, y0)`                | `w[0:2]` of any sample (any dataset) | ✅ directly                           |
| `r0`                      | `sqrt(x0² + y0²)`                    | ✅ derived                            |
| `θ(λ)`                    | `λ · atan2(y0, x0)`                  | ✅ for each λ in `_LAMBDA_SET`        |
| `s(λ)`                    | `exp(λ · tanh(r0))`                  | ✅ for each λ in `_LAMBDA_SET`        |
| `P_uniform(λ)`            | `1 / |Λ|`                            | ✅ flat 20 % per λ                    |
| `P_softmax(λ)`            | `softmax(α)`                         | ✅ if `α` is published                |
| `λ` per sample            | not stored                           | ❌ must be inferred                   |
| `p_k, k ≥ 1` (clean)      | `(1/s) · R(−θ) · w_k`                | exact once λ is known                 |
| `p_k, k ≥ 1` (noisy)      | `(1/s) · R(−θ) · w_k`                | approximate; ±5 % per element bound;  |
|                           |                                      | average over a λ-cluster to denoise   |
| Noise model               | `x · N(1, σ²)`, clipped to ±5 %·\|x\| | ✅ if `σ` is published (default 0.02) |
| Which dataset uses what   | raw / noisy: uniform                 | softmax: weighted / weighted_noisy    |
