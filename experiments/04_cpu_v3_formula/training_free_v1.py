"""Exploratory (not pre-registered): use the training-free score  z(OV path at init) + z(D)  on the v1 data.

(a) v1 held-out test factorial (16 inits x 16 splits): top-3 hits against the step-500 winners, compared with the
    OV-path init readout alone (the v1 R1 baseline) and with C4 (which needed 64 extra trained runs per split).
(b) v1 full runs (64 runs trained to the end): how many of the top-3 predicted frequencies are in the final circuit.
Writes results/training_free_v1.json.
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(os.path.dirname(_HERE))
os.environ["GROK_OUT"] = os.path.join(_ROOT, "experiments", "02_kaggle_v1", "results")
sys.path.insert(0, os.path.join(_ROOT, "experiments", "02_kaggle_v1")); sys.path.insert(0, os.path.join(_ROOT, "src"))
import numpy as np, torch
import gpu_grok as G
from freqsel.tasks import Task
from freqsel.predictors import label_sharing_D
from freqsel.models import init_state
from freqsel.spectra import path_share

T = Task("add", 113)
I, S, E, A0, _ = G.factorial_arrays("B_test", G.WIN_STEP)
W3 = np.argsort(-E, -1)[..., :3]
D = np.stack([label_sharing_D(T, s) for s in S])                      # [S, 56] (log D)
A = np.log(A0)
score = G.zz(A) + G.zz(D)[None]
Hc = G.topk_hits_cells(score, W3); Ha = G.topk_hits_cells(G.zz(A), W3)
Hd = G.topk_hits_cells(np.broadcast_to(G.zz(D)[None], E.shape), W3)
n = len(I) * len(S)
res = dict(test_factorial=dict(hits_training_free=int(Hc.sum()), hits_attention_only=int(Ha.sum()),
                               hits_D_only=int(Hd.sum()), of=3 * n, chance=n * 9 / 56,
                               ci_diff=G.paired_boot_IS(Hc, Ha),
                               C4_hits_needing_trained_votes=json.load(open(os.path.join(G.OUT, "F_q1_results.json")))["C4"]["hits_combined"]))

runs = json.load(open(os.path.join(G.OUT, "C_chunk_0000.json")))
prec, base, prec_a = [], [], []
for r in runs:
    W = {k: v[None] for k, v in init_state(r["init"]).items()}
    a = np.log(path_share(W, T)[0].numpy()); d = label_sharing_D(T, r["split"])
    keys0 = {k - 1 for k in r["final_keys"]}
    top = np.argsort(-(G.zz(a) + G.zz(d)))[:3]; top_a = np.argsort(-a)[:3]
    prec.append(len(keys0 & set(top.tolist())) / 3); prec_a.append(len(keys0 & set(top_a.tolist())) / 3)
    base.append(len(keys0) / 56)
res["final_circuit"] = dict(runs=len(runs), top3_precision_training_free=float(np.mean(prec)),
                            top3_precision_attention_only=float(np.mean(prec_a)), chance=float(np.mean(base)))
json.dump(res, open(os.path.join(_HERE, "results", "training_free_v1.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
