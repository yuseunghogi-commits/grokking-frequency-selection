# Technical write-up

*Claude (Anthropic) with Seungho Yu, October 2026. Not peer reviewed.*

## 1. Setup

**Task and model.** We train 1-layer transformers on `(a, b, =) → (a ∘ b) mod p` with ∘ ∈ {+, −, ×}. The model has
d = 128, 4 heads, d_mlp = 512, ReLU, no LayerNorm, untied embed and unembed, and follows Nanda et al. 2023. Training uses
full-batch AdamW (lr 1e-3 with 10-step linear warm-up, β = (0.9, 0.98), weight decay 1.0) on a random fraction (default
30 %) of all p² pairs. An *init* is a torch seed for the parameters. A *split* is a seed for the train/test partition, and
the two are varied independently. For many runs in parallel we stack M models and train them with einsum. A custom
StackedAdamW reproduces `torch.optim.AdamW` exactly (max |Δ| in spectra 1e-5 on GPU and 0.0 on CPU).

**Readouts.** The *embedding spectrum* of a run is the share of the Fourier power of W_E (over tokens) in each
frequency k = 1..⌊(n−1)/2⌋. For multiplication, tokens are re-indexed by their discrete log (primitive root g), so the
group is Z_{p−1} and n = p − 1. The *key frequencies* of a trained run are found by clustering MLP neurons by their dominant
frequency and keeping clusters that hold ≥ 3 % of the live neurons. We checked them causally: restricting the logits to
the key-frequency components keeps test accuracy at 100 %, and removing them takes it to chance.
The *winner proxy* is the top-3 frequencies of the embedding spectrum at step 500, or at step 100 to 200 in later
experiments.

**Votes.** For a factorial grid of inits × splits, each cell has a z-scored log spectrum Z[i,s,k]. Two votes are defined
for each cell, both computed without using that cell:

- the *init vote* is the mean of Z over the same init on other splits;
- the *data vote* is the mean of Z over the same split with other inits.

From v2 onwards, the data vote is centred across splits, which removes the component that all splits share because of
shared inits. Conditions in v2 use disjoint init ranges for the same reason: in a smoke test, a shared-init artefact
had produced a spurious agreement of 0.98.

**Statistics.** The main metric is top-3 hits: |predicted top-3 ∩ winner top-3| summed over cells. Chance is 9 / n_f per
cell. The null distributions are cluster-aware permutations: frequencies are permuted within splits or within inits,
because a hypergeometric null overstated significance roughly 300-fold in a smoke test. We also use paired bootstraps
over inits and splits, within-split Spearman correlations with permutation p-values, Holm correction, and Spearman–Brown
split-half reliability as a noise ceiling.

## 2. Experiments

| | where | runs | pre-registered claims |
|---|---|---|---|
| pilot | CPU | ~60 single runs | grokking reproduced, frequencies vary with seed, early spectrum predicts final keys |
| v1 | Kaggle T4, 9.4 h | 2 factorials (dev / test, 16 × 16), 64 full runs, 2 × 1536-run datamodels, MLP and subtraction variants | C1–C6, R1 |
| v2 | Kaggle T4 | shift / shuffle / subtraction / MLP conditions, datamodels on 4 splits, time course | H1–H5 |
| v3 | CPU | 16 splits × 16 inits, 200 steps | P1, P2 (D(k)) |
| derivation | CPU | ΔE vs D per run; training fraction 0.15–0.7 | exploratory |
| v4 | CPU | 4 configs × 8 splits × 12 inits, 100 steps | G1, G2 |

## 3. Results

### 3.1 Init and data both vote, roughly additively (v1)

On the held-out test factorial, top-3 hits out of 3072 were 655 for the init vote, 959 for the data vote and 1265 for
their sum. Chance is 165. Both bootstrap CIs of the improvement exclude zero (C1, PASS). The step-500 proxy agrees with
the step-200 proxy in 96 % of cases. It contains only 56.5 % of the final key frequencies, though, so it is a
leading-frequency target and not the final circuit (the proxy validity criterion failed).

The data effect is causal (C3). A ridge datamodel was fitted on 1536 runs per split to rank training pairs by their
effect on the split's favourite frequency k\*. Removing the top 10 % of pairs pushed z(k\*) to −0.71 and −1.01, against
+1.19 and +1.30 for random 10 % removals and +3.86 and +3.47 for removing the bottom 10 %. Holm p = 0.0002 on both splits.

The init readout works through the attention path (R1, PASS). The Fourier power of W_E·W_V·W_O at initialisation, before
any training, predicts the init vote. Combining it with an independently trained data vote gave 1256 hits vs 576
(C4, PASS).

Failures: C2, a search over simple data statistics, did not transfer to test. C6, predicting the time of grokking from
internal measures, lost to a behavioural baseline (C 0.85 vs 0.92).

### 3.2 What casts the data vote (v2)

Agreement with the true data vote, against a noise ceiling of about 0.95:

- shift relabelling a → a + t: 0.955;
- shuffled labels: 0.04;
- subtraction on the same pairs: 0.006;
- an MLP-only architecture on the same splits: 0.69.

The vote therefore depends on the algebraic structure of which equations are present, not on token identities. It is
also mostly a property of the data, not of attention. Per-pair datamodels predict it poorly held out (R² ≈ 0.03; H5
failed), so it is a collective property of the split. The time course shows that the data vote appears in the embedding
and the OV path at steps 8–15 (agreement 0.56 by step 16 and 0.85 by step 32). The unembedding lags behind and only
reaches 0.6 by step 96.

### 3.3 A closed form: D(k) (v3)

Two inputs a and a′ *share an answer* c when both (a, c−a) and (a′, c−a′) are in the training set. Let N[a,a′] count
the shared answers. Its Fourier power is

  D(k) = Σ_c | Σ_{(a, c−a) ∈ train} e^{2πika/p} |² ,

summed over both operand positions. D has no free parameters. In the pre-registered test (16 fresh splits × 16 fresh
inits):

- **P1:** within-split Spearman(D, data vote) = 0.709, permutation p = 1e-4, against a vote reliability of 0.93. PASS.
- **P2:** the score z(OV path at init) + z(D) gave 230 top-3 hits of 768, vs 154 for the OV path alone and 135 for D
  alone (chance 41; CI of the difference [28, 123]). PASS.

Exploratory, on the v1 data: the same score, with no training at all, gets 961 hits on the v1 test factorial vs 576 for
the init readout alone. C4 needed 64 extra trained runs per split and reached 1256. Scored against the final circuits of
the 64 fully trained v1 runs, 43 % of its top-3 frequencies are key frequencies, vs 27 % for the init readout alone and
9 % chance (`experiments/04_cpu_v3_formula/training_free_v1.py`).

### 3.4 Why D(k)

At initialisation the logits are nearly uniform, so the gradient of the cross-entropy with respect to the input
embedding E_a is approximately −Σ_{b:(a,b)∈train} W_U[:, a∘b] (with the path through attention and the MLP linearised).
With random, nearly orthogonal unembedding columns r_c, this gives

  ΔE_a ≈ η Σ_c N₁[a,c] r_c   ⇒   ΔE ΔEᵀ ≈ η² N ,

where N₁[a,c] marks whether a appears with answer c. The Fourier power of the early embedding update at frequency k is
therefore proportional to D(k). This is the leading term of known early-dynamics analyses (Yao & Xu 2025; Wang et al.
2026; Tian 2025), so we do not claim it as new. What is new is applying it to a *finite split*, where N is not
circulant, and testing it per frequency.

Checks (exploratory):

- Each run's own ΔE spectrum correlates with its own split's D (ρ 0.33–0.46). It anti-correlates with another split's D.
- The training-fraction sweep gave ρ = 0.65, 0.73, 0.64 and 0.65 at fractions 0.15, 0.3, 0.5 and 0.7.
- The relative spread of D over k was 0.095–0.099, vs ≈ 1/√p = 0.094 predicted from random-set statistics.
- The data vote weakens as the fraction grows (0.59 → 0.36 at 0.7). With more data, N approaches circulant, D flattens
  and the init dominates.

### 3.5 Generalisation (v4)

All four configurations pass both tests: Spearman ρ = 0.645, 0.739, 0.673 and 0.690 for add 59, mul 59, add 97 and mul 97
(all p = 1e-4), and spread ratios 0.89, 1.01, 0.93 and 0.92. Vote reliability is 0.93–0.94 throughout (pre-registered G1: ρ > 0.3 with p < 0.01 in every configuration; G2: spread within ±25 %
of 1/√n). For multiplication, the predictor is computed in the discrete-log basis. Its success is evidence that the
mechanism depends on the group structure and not on the particular way the tokens are encoded.

### 3.6 Steering: choosing the frequency in advance (v5)

If D(k) describes how the split shapes selection, it should work as a control knob. We drew 8 target frequencies at
random (k\* ∈ {2, 17, 25, 28, 33, 35, 41, 43}). For each one we took a random base split and greedily swapped training
pairs with held-out pairs to raise J = log D(k\*) − mean_{k≠k\*} log D(k), at 5 strengths. Neither the split size nor the
per-input or per-answer pair counts changed, and no network was involved. At the strongest level, 412–481 pairs (11–13 %; the
pre-registration said "about 430–470") were swapped. Twelve fresh inits were trained on every split.

| | original | g = 0.15 | 0.3 | 0.6 | 1.0 |
|---|---|---|---|---|---|
| pairs swapped (mean) | 0 | 57 | 114 | 243 | 436 |
| target in step-200 top-3 | 4 % | 11 % | 42 % | 96 % | **100 % (96/96)** |

- **S1:** rate 1.0 ≥ 0.5, and the improvement over the original split is positive for every target (sign-flip p = 0.004,
  the minimum possible with 8 targets). **PASS.**
- **S2:** mean within-target Spearman(strength, rate) = 0.95, p = 1e-4. **PASS.**
- **S3:** after 10 000 steps, the target is a key frequency of the final circuit in **16/16** steered runs, vs 1/16 for the
  same inits on the original splits (Fisher p = 2.8·10⁻⁸). **PASS.**
- **S4 (descriptive):** 15/16 steered runs grokked (median step 4250), vs 16/16 on the original splits (median 6375).
  Mean final test accuracy was 0.992 vs 1.000. The run that did not grok reached 0.88.
- Exploratory: steering worked equally well whether the init ranked k\* in the top or the bottom half of its OV-path
  spectrum (100 % in both).

Stage B ran its first 4 runs on CPU. The remaining 28 ran on a Kaggle T4 GPU, using the same code and the same
hash-verified splits; the device was not specified in the pre-registration.

### 3.7 A failed law for the training fraction (v6)

The data vote weakens as the training fraction grows. An energy-fraction argument predicts that its amplitude relative
to the init's scales as √(1 − f). The pre-registered test at f = 0.85 gave a ratio of 0.29 against the f = 0.3 value.
The prediction was 0.46 with a window of 0.32–0.60, so the test **failed**. The effect shrinks faster than predicted,
roughly as (1 − f)^0.7–0.8. See docs/theory_init_side.md, which also covers the exploratory init-side theory: the OV-path
argument, the output-side product rule, and the failed power-sum model.

## 4. Limitations

- Everything is small: 1-layer models, p ≤ 113, a single optimiser setting with frac 0.3 by default. Whether anything
  similar happens in larger models or on non-group tasks is open.
- Most predictions target the leading frequencies early in training, and these overlap only partly (≈ 56 %) with the
  final circuit. The final-circuit check (43 % vs 9 %) is exploratory.
- D(k) explains the data side. The init side has a partial explanation (OV path; output-side product rule), but the
  relative weight of init and data is not derived.
- The literature search was done with tools and may be incomplete.
- One AI system designed, ran and analysed these experiments. Pre-registration and the released code are meant to make
  them checkable, but independent replication has not been done.
