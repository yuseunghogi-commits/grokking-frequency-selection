# The init side: what the initialisation contributes, and why

*Exploratory analysis (not pre-registered), run on stored v1 and v3 data. Script: `experiments/08_init_theory/init_side.py`;
numbers: `experiments/08_init_theory/results/init_side.json`.*

The data side has a closed form, D(k). Until now the init side was an empirical readout: the Fourier power of
W_E·W_V·W_O at initialisation. This note asks why that readout works and whether it can be improved from first principles.

## T1. The MLP sees tokens only through the OV path

At the `=` position the residual stream contains E_= + pos₂ + Σ_j α_j (E_{x_j} + pos_j) W_OV, followed by the MLP. The
operands a and b never enter that position's residual directly; they arrive only through the attention OV map. Grokking
builds Fourier features in the MLP, so the "lottery ticket" amplitude that frequency k starts with should be that of
the token map **as seen through OV**, not that of W_E itself. MLP-only models are different: there the MLP reads E
directly, and Ding et al. (2024) report that the W_E spectrum predicts survival.

| init predictor | v1 dev hits | v1 test hits | v3 hits | ρ with init vote (dev / test / v3) |
|---|---|---|---|---|
| Fourier power of W_E | 523 | 485 | 132 | 0.67 / 0.69 / 0.71 |
| Fourier power of W_E·W_OV | **594** | **576** | **154** | **0.77 / 0.77 / 0.72** |

Chance is 165 per 3072 (v1) and 41 per 768 (v3). The OV readout wins in all three data sets, as predicted.

## T2. Output amplitude matters too: a parameter-free product rule

A frequency-k feature is a product of an input factor (tokens → MLP at frequency k) and an output factor (MLP →
answer classes at frequency k). For a two-factor feature u_k·v_k under gradient flow from small initialisation, the escape
time is ≈ −log(u_k v_k)/s. The predicted ranking is therefore log P_in(k) + log P_out(k), where P_out(k) is the Fourier
power of W_U over answer classes. This rule has no free parameter.

| | v1 dev | v1 test | v3 |
|---|---|---|---|
| W_U spectrum alone | 330 (ρ 0.29) | 363 (ρ 0.26) | 83 (ρ 0.15) |
| log P_in (OV) | 594 | 576 | 154 |
| **log P_in + log P_out** (no fitting) | **680** (ρ 0.80) | **646** (ρ 0.79) | **162** (ρ 0.73) |
| log P_in + 0.5 log P_out (best scanned weight) | 669 (ρ 0.82) | 625 (ρ 0.81) | 173 (ρ 0.75) |

The unembedding carries independent information about which frequency will win. This information is present at
initialisation, before any data is seen. The parameter-free product is close to the best weight in the scan.
Combined with D(k) (equal-weight z-sum, the same rule as v3), training-free prediction improves on every data set:
945 → 992 (dev), 961 → **1008** (test), 230 → 234 (v3). Precision against the final circuits of the 64 fully trained v1
runs rises from 0.43 to 0.45, with chance at 0.09. These gains are consistent in sign but small.

## T3. The simplest unified model fails, and that is informative

Suppose the data side were an extra random-direction drive whose power at frequency k is D(k). The effective input power
would then be P_in(k) + γ·D(k). The best γ on dev gives 829 hits, far below the equal-weight z-sum (945). The test and v3
sets show the same gap (824–834 vs 961; 199 vs 230). D varies across k by only ≈ 1/√p ≈ 10 %, so adding it as power
barely changes the ranking. In real training its influence is much larger.

Our interpretation, not yet tested directly: the data-driven update ΔE_a ≈ η Σ_c N[a,c] W_OV W_U[:,c] is assembled from
the output pathway's own columns, so it starts **aligned** with the output factor. A random initial embedding at frequency
k is aligned with the output factor only by chance (cos θ ~ 1/√d). An aligned seed counts for more in the u·v growth by
roughly √d, which would explain why the data's small relative variation has an outsized effect.

## T4. One exact gradient at init

Neither training nor updates are needed. The Fourier power of the exact loss gradient with respect to W_E at
initialisation, on the actual training split, predicts the winners by itself (749 v1 test, 139 v3). Combined with the
OV readout it matches the closed-form rule (1002 vs 961; 236 vs 230). This fits D(k) being the leading term of that gradient.

## The fraction dependence (open question from report section 12)

As the training fraction f rises, the data vote weakens. With mean-reduced loss, the fraction of the embedding update's
energy that lies at any nonzero frequency equals (1 − f)/p. This follows by Parseval from E|S_c(k)|² = n_c(1 − f) under
sampling without replacement. The data drive's amplitude relative to the init's should therefore scale as √(1 − f).
Noise-corrected √(data variance / init variance), relative to f = 0.3:

| f | 0.15 | 0.3 | 0.5 | 0.7 |
|---|---|---|---|---|
| observed | 1.08 | 1.00 | 0.90 | 0.57 |
| √(1 − f) law | 1.10 | 1.00 | 0.85 | 0.66 |

The law captures the trend but was derived after seeing these data.

**Pre-registered test (PREREG6, f = 0.85, fresh splits and inits): FAILED.** The observed ratio r(0.85)/r(0.3) is
R = 0.289 (bootstrap 95 % interval 0.30–0.42). The pre-registered window was [0.324, 0.602] around the √(1 − f) prediction
of 0.463. The data vote shrinks faster than √(1 − f): R lies between the square-root law (0.463) and the linear law
(0.214), and closer to the linear one. Fitting R = ((1 − f)/0.7)^α gives α ≈ 0.80 here and ≈ 0.67 at f = 0.7 in the
earlier sweep (exploratory). The energy-fraction argument therefore gets the direction right but not the size. A
missing ingredient, for example the shrinking per-pair gradient weight 1/(f p²) interacting with Adam's normalisation,
is left open.

## What is still open

- A derivation of the alignment argument in T3 that gives the relative weight of the init and data terms. The equal-weight
  z-sum works empirically, but it is not derived.
- Why output-side information helps more at step 500 (v1) than at step 200 (v3).
