# `transformation.py` — Full Documentation

Source: `src/transformation.py`

This document is written for someone who has been handed the generated
dataset and must **recover the hidden ground-truth vector `v`**. It
explains the transform in enough detail that the inverse problem
becomes tractable.

---

## 1. Overview

The module implements a nonlinear, causal transformation on a vector

```
v = [x0, y0, x1, y1, x2, y2, ..., x_{n-1}, y_{n-1}]
```

interpreted as `n` 2D coordinate pairs
`p_0, p_1, ..., p_{n-1}` where `p_k = (x_k, y_k)`.

A hidden scalar parameter **λ** (drawn from a small discrete set
`_LAMBDA_SET`) controls the transform. Each run of
`generate_dataset` produces many outputs `w = T(v, λ_i)` — one for
each sampled λ. The user's job is to recover `v` from the collection
of `w`s without knowing which λ produced each one.

---

## 2. The Transform, Step by Step

### 2.1 Anchor

The first pair `p_0 = (x0, y0)` is the **anchor**. It has two roles:

1. It is copied into the output unchanged: `w_0 = p_0`.
2. It alone determines the rotation angle and scale factor applied
   to every other pair.

> **Key consequence:** every output vector `w` begins with the exact
> same `(x0, y0)` as the input `v`. The first pair of the hidden
> vector is therefore **not hidden at all** — it is visible in every
> sample of the dataset.

### 2.2 Derived quantities

From the anchor the transform computes three scalars:

```
r0   = sqrt(x0² + y0²)              # anchor magnitude
θ(λ) = sign(λ) · atan2(y0, x0)      # rotation angle
s(λ) = exp(λ · r0)                  # scale factor
```

Note:

- `r0` depends only on `v`, not on λ.
- `θ(λ)` flips sign with λ; when `λ = 0` the angle collapses to `0`.
- `s(λ)` is strictly positive; `s(0) = 1` and `s(-λ) = 1 / s(λ)`.

### 2.3 Per-pair update

For every `k ≥ 1`:

```
w_k = s(λ) · R(θ(λ)) · p_k
```

where `R(θ)` is the standard 2D rotation matrix

```
R(θ) = [ cos θ   -sin θ ]
       [ sin θ    cos θ ]
```

Written out in components:

```
w_k.x = s · (x_k · cos θ − y_k · sin θ)
w_k.y = s · (x_k · sin θ + y_k · cos θ)
```

**The same `s` and `θ` are applied to every non-anchor pair inside a
single output `w`.** The transform is globally consistent within a
sample — it is not pair-specific.

---

## 3. Worked Micro-Example

Take `v = [3, 4, 1, 0, 0, 2]` and `λ = 0.1`.

```
r0 = sqrt(3² + 4²) = 5
θ  = sign(0.1) · atan2(4, 3) ≈ +0.9273 rad
s  = exp(0.1 · 5) = exp(0.5) ≈ 1.6487
```

Anchor copied: `w_0 = (3, 4)`.

Pair `p_1 = (1, 0)`:

```
w_1.x = 1.6487 · (1 · cos 0.9273 − 0 · sin 0.9273) ≈ 0.989
w_1.y = 1.6487 · (1 · sin 0.9273 + 0 · cos 0.9273) ≈ 1.319
```

Pair `p_2 = (0, 2)`:

```
w_2.x = 1.6487 · (0 · cos 0.9273 − 2 · sin 0.9273) ≈ −2.639
w_2.y = 1.6487 · (0 · sin 0.9273 + 2 · cos 0.9273) ≈ 1.979
```

Output: `w ≈ [3, 4, 0.989, 1.319, −2.639, 1.979]`.

---

## 4. Key Properties (Useful for Inversion)

1. **Anchor preservation.**
   `w[0:2] == v[0:2]` for every sample and every λ.

2. **Exact invertibility given λ.**
   Because `s(−λ) = 1/s(λ)` and `θ(−λ) = −θ(λ)`, applying the
   transform again with `−λ` undoes it:
   `transform(transform(v, λ), −λ) == v`.

3. **λ = 0 is the identity.**
   If λ = 0 then `s = 1` and `θ = 0`, so `w == v`. Samples produced
   with λ = 0 reveal the ground truth directly.

4. **Finite hidden set.**
   `_LAMBDA_SET = [0.1, −0.05, 0, 0.05, 0.1]` — only **four distinct
   values** (0.1 is duplicated, so that λ is twice as likely). Every
   sample in the dataset therefore comes from one of four
   (`s`, `θ`) pairs.

5. **Uniform scaling inside a sample.**
   For any two non-anchor pairs `p_j, p_k` inside the **same**
   output `w`:
   `||w_j|| / ||w_k|| == ||p_j|| / ||p_k||`
   and the angle between `w_j` and `w_k` equals the angle between
   `p_j` and `p_k`. The transform rotates and rescales them
   **together**, so their relative geometry is preserved.

6. **Cross-sample ratios reveal `s`.**
   For two samples `w^{(a)}` and `w^{(b)}` of the same pair index
   `k ≥ 1`:
   `||w_k^{(a)}|| / ||w_k^{(b)}|| == s(λ_a) / s(λ_b)`.
   This ratio does not depend on `p_k` at all.

---

## 5. Recovery Strategy

Given a dataset of `N` samples `{w^{(i)}}`, a clean path to `v` is:

### Step A — Read the anchor directly.

`(x0, y0) = w^{(i)}[0:2]` for **any** `i`. Compute `r0 = sqrt(x0² + y0²)`.

### Step B — Enumerate the four candidate `(s, θ)` pairs.

Since `_LAMBDA_SET` has four distinct values and `r0, x0, y0` are
known, you can compute all four candidate `(s(λ), θ(λ))` pairs
explicitly — even though λ itself is "hidden", the finite set of
possible transforms it induces is fully determined by the anchor.

### Step C — Classify each sample.

For each sample `w^{(i)}`, pick any `k ≥ 1` and compute

```
p_k^{(i, guess)} = (1 / s_guess) · R(−θ_guess) · w_k^{(i)}
```

for each of the four candidate `(s_guess, θ_guess)` pairs. The
correct λ is the one that makes the recovered `p_k` **consistent
across all samples** (they should all produce the same point for the
same underlying `p_k`). Cluster samples by this consistency.

### Step D — Invert.

Once each sample is labeled with its λ, apply `transform(w, −λ)` to
recover `v` from any single sample. With multiple samples you can
average to suppress floating-point noise.

### Shortcut.

Any sample whose non-anchor pairs have the same magnitudes as the
anchor-scaled identity (i.e. `s = 1`, `θ = 0`) was generated with
λ = 0 and **equals `v` directly**. Because λ = 0 is in the hidden
set, such samples are guaranteed to exist in a large enough dataset.

---

## 6. API Reference

### `transform(v, lam)`

Apply the anchor-based nonlinear rotation-scaling transform.

| Parameter | Type                             | Description                                      |
|-----------|----------------------------------|--------------------------------------------------|
| `v`       | `numpy.ndarray` of shape `(2n,)` | Input vector of `n` 2D coordinate pairs.         |
| `lam`     | `float`                          | Transformation parameter λ. Pass `−λ` to invert. |

Returns `w : numpy.ndarray` of the same shape as `v`.

Behaviour:

- `w[0:2] = v[0:2]` (anchor is copied).
- For `k ≥ 1`, `w[2k:2k+2]` is `s · R(θ)` applied to `v[2k:2k+2]`.

---

### `generate_dataset(v, n_samples=400)`

Generate multiple transformed outputs from the same input vector.
Each sample uses a λ drawn uniformly from `_LAMBDA_SET` via
`np.random.choice`.

| Parameter   | Type        | Description                     |
|-------------|-------------|---------------------------------|
| `v`         | array-like  | Ground truth vector.            |
| `n_samples` | `int`       | Number of transformed outputs.  |

Returns a `(n_samples, len(v))` NumPy array. The λ value used for
each row is **not** stored — recovering it is part of the challenge.

---

## 7. Module-Level Script Behaviour

Running `python src/transformation.py` executes the following flow:

1. **Prompt** the user for a ground-truth vector, parsed with `eval`.
   The vector must have even length; it is never echoed to stdout.
2. **Generate** 400 transformed outputs via `generate_dataset`.
3. **Reversibility check**: for each λ in `_LAMBDA_SET`, confirm that
   `transform(transform(v, λ), −λ) == v` using `np.allclose` with
   `atol = 1e-10`. Only pass/fail is printed.
4. **Save** the dataset to
   `C:\Users\DELL\Documents\GitHub\vector_Transformations\data\transformed_dataset.npy`.
5. **Reload** the saved `.npy`, print its shape and the first five
   rows rounded to three decimals.
6. **Dump** every row of the dataset (rounded) so a solver can
   inspect the full collection.

The ground-truth vector `v` and the per-sample λ are never printed.

---

## 8. Dependencies

- `numpy`
- `os` (standard library)

---

## 9. Summary Cheat-Sheet

| Quantity      | Formula                              | Known to solver? |
|---------------|--------------------------------------|------------------|
| `(x0, y0)`    | `w[0:2]` of any sample               | ✅ directly       |
| `r0`          | `sqrt(x0² + y0²)`                    | ✅ derived        |
| `θ(λ)`        | `sign(λ) · atan2(y0, x0)`            | ✅ for each λ in `_LAMBDA_SET` |
| `s(λ)`        | `exp(λ · r0)`                        | ✅ for each λ in `_LAMBDA_SET` |
| `λ` per sample| not stored                           | ❌ must be inferred |
| `p_k, k≥1`    | `(1/s) · R(−θ) · w_k`                | recoverable once λ is known |
