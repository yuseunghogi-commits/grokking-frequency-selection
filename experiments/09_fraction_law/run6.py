"""PREREG6 runs: fractions 0.3 and 0.85, fresh splits 7600-7607 and inits 7700-7711, 60 steps (as the v4-era sweep). Resumable."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.environ["GROK_OUT"] = os.path.join(HERE, "results"); sys.path.insert(0, os.path.join(ROOT, "experiments", "02_kaggle_v1"))
import numpy as np, torch, gpu_grok as G
torch.set_num_threads(2)
out = G.OUT / "runs"; out.mkdir(parents=True, exist_ok=True)
SPLITS = list(range(7600, 7608)); INITS = list(range(7700, 7712))
for frac in (0.3, 0.85):
    for s in SPLITS:
        f = out / f"f{frac}_s{s}.npz"
        if f.exists(): continue
        E = [G.train_batch([dict(init=i, split=s) for i in INITS[c0:c0 + 4]], 60, record=(60,), frac=frac)["rec"][60]["embed"]
             for c0 in range(0, 12, 4)]
        np.savez(f, embed60=np.concatenate(E)); print(frac, s, flush=True)
print("ALLDONE", flush=True)
