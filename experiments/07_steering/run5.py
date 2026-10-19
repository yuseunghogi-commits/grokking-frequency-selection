"""v5 steering runs (CPU). Resumable. Stage A: 8 targets x 5 levels x 12 inits, 200 steps.
Stage B: strongest level and base split (level 0), 2 inits per target, trained 10000 steps with final circuit analysis."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.environ["GROK_OUT"] = os.path.join(HERE, "results"); sys.path.insert(0, os.path.join(ROOT, "experiments", "02_kaggle_v1"))
import numpy as np, torch
import gpu_grok as G
torch.set_num_threads(2)
S = np.load(os.path.join(HERE, "splits.npz")); I = json.load(open(os.path.join(HERE, "splits_info.json")))
TARGETS, LEVELS, BASE = I["targets"], I["levels"], {int(k): v for k, v in I["base"].items()}
INITS = list(range(9600, 9612)); FULL_INITS = INITS[:2]
outA = G.OUT / "A"; outA.mkdir(parents=True, exist_ok=True); outB = G.OUT / "B"; outB.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for k in TARGETS:
    for g in LEVELS:
        for c0 in range(0, 12, 4):
            f = outA / f"k{k}_g{g}_c{c0}.npz"
            if f.exists(): continue
            ch = INITS[c0:c0 + 4]; tr = torch.as_tensor(S[f"k{k}_g{g}"])
            r = G.train_batch([dict(init=i, split=BASE[k]) for i in ch], 200, record=(0, 200), custom_train_idx=[tr] * 4)
            np.savez(f, inits=np.array(ch), embed200=r["rec"][200]["embed"], path0=r["rec"][0]["path"])
            print(f"A k{k} g{g} c{c0} {time.time()-t0:.0f}s", flush=True)
print("A_DONE", flush=True)
jobs = [(k, g, i) for k in TARGETS for g in (LEVELS[-1], 0.0) for i in FULL_INITS]
for j0 in range(0, len(jobs), 4):
    f = outB / f"b{j0:02d}.json"
    if f.exists(): continue
    js = jobs[j0:j0 + 4]
    r = G.train_batch([dict(init=i, split=BASE[k]) for k, g, i in js], 10000, record=(200,), eval_every=250,
                      custom_train_idx=[torch.as_tensor(S[f"k{k}_g{g}"]) for k, g, i in js])
    rows = [dict(k=k, level=g, init=i, grok=r["grok"][m], final=r["final"][m],
                 top3_200=np.argsort(-r["rec"][200]["embed"][m])[:3].tolist()) for m, (k, g, i) in enumerate(js)]
    f.write_text(json.dumps(rows, default=float)); print(f"B {j0} {time.time()-t0:.0f}s", flush=True)
print("ALLDONE", flush=True)
