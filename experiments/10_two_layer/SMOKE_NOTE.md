# Disclosure: pipeline smoke test overlapped with confirmatory seeds

After PREREG7 was written, hashed and pushed (commit before any v7 run), the Kaggle notebook was smoke-tested on CPU
with `V7_SMOKE=1`. This used a subset of the confirmatory configuration: targets 2 and 17, inits 9800-9803, and stage B
cut to 300 steps. The smoke output was seen before the main run:

- T2 on 2 splits: Spearman 0.63;
- T4 on 4 inits: E 0.60, OV1 0.50, ALL 0.49 (ALL - OV1 ≈ 0);
- no run grokked in 300 steps, as expected.

Nothing in PREREG7, the runner or the analysis was changed after seeing these numbers. The main run on the Kaggle GPU
recomputes all of them, so the smoke runs are not reused. The T4 hint (ALL not better than OV1) is recorded here in
advance of the main result.
