"""Figure: centred D(k) vs centred data vote for the four v4 configs -> report/figures/v5_generalise.png"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import rankdata
from freqsel.tasks import Task
from freqsel.predictors import label_sharing_D
from freqsel.stats import zscore_log
RES = os.path.join(ROOT, "results", "v4"); SPLITS = list(range(8000, 8008))
R = json.load(open(os.path.join(RES, "V4_results.json")))
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for j, (name, p) in enumerate([("add", 59), ("mul", 59), ("add", 97), ("mul", 97)]):
    T = Task(name, p); tag = f"{name}{p}"
    Z = np.stack([zscore_log(np.load(os.path.join(RES, f"{tag}_s{s}.npz"))["embed100"]) for s in SPLITS])
    V = Z.mean(1); V -= V.mean(0, keepdims=True)
    D = np.stack([label_sharing_D(T, s) for s in SPLITS]); D -= D.mean(0, keepdims=True)
    for i in range(len(SPLITS)):
        ax[j].scatter(rankdata(D[i]) / D.shape[1], rankdata(V[i]) / D.shape[1], s=8, alpha=.5)
    r = R[tag]
    ax[j].set_title(f"{'a+b' if name == 'add' else 'a·b'} mod {p}:  ρ = {r['spearman']:.2f}")
    ax[j].set_xlabel("rank of D(k) within split"); ax[j].set_aspect("equal")
ax[0].set_ylabel("rank of data vote within split")
fig.suptitle("Pre-registered v4: the closed-form D(k) predicts the data vote at other primes and for multiplication")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "report", "figures", "v5_generalise.png"), dpi=110)
