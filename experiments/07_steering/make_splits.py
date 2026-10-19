"""Build the v5 steered splits (deterministic). Writes splits.npz and splits_info.json."""
import json, os, numpy as np
from steer import steer, split_perm, D_of, objective
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(5)
TARGETS = sorted((rng.choice(56, 8, replace=False) + 1).tolist())          # 1-based frequencies
BASE = {k: 9500 + j for j, k in enumerate(TARGETS)}
LEVELS = [0.0, 0.15, 0.3, 0.6, 1.0]
out, info = {}, []
for k in TARGETS:
    for g in LEVELS:
        if g == 0: tr = np.sort(split_perm(BASE[k])[0]); log = dict(gain=0.0, accepted=0)
        else: tr, log = steer(BASE[k], k - 1, g, rng_seed=k)
        D = np.log(D_of(tr)); z = float((D[k - 1] - D.mean()) / D.std())
        base = set(split_perm(BASE[k])[0].tolist())
        out[f"k{k}_g{g}"] = tr
        info.append(dict(k=k, level=g, base_seed=BASE[k], z_target=z, rank_target=int((D > D[k - 1]).sum()),
                         swapped=int(len(set(tr.tolist()) - base)), **{kk: v for kk, v in log.items() if kk != "J0"}))
        print(info[-1], flush=True)
np.savez(os.path.join(HERE, "splits.npz"), **out)
json.dump(dict(targets=TARGETS, base=BASE, levels=LEVELS, splits=info), open(os.path.join(HERE, "splits_info.json"), "w"), indent=1)
