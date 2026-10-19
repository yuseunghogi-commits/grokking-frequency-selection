import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(os.path.dirname(_HERE))
os.environ.setdefault("GROK_OUT", os.path.join(_HERE, "results")); sys.path.insert(0, os.path.join(_ROOT, "experiments", "02_kaggle_v1"))
import numpy as np, torch, gpu_grok as G
torch.set_num_threads(2)
out = G.OUT / "frac"; out.mkdir(parents=True, exist_ok=True)
SPLITS = list(range(7200, 7208)); INITS = list(range(7300, 7312))
for frac in (0.3, 0.15, 0.5, 0.7):
    for s in SPLITS:
        f = out / f"f{frac}_s{s}.npz"
        if f.exists(): continue
        E = []
        for c0 in range(0, 12, 4):
            r = G.train_batch([dict(init=i, split=s) for i in INITS[c0:c0+4]], 60, record=(0, 60), frac=frac)
            E.append(r["rec"][60]["embed"])
        np.savez(f, embed60=np.concatenate(E)); print(frac, s, flush=True)
print("ALLDONE", flush=True)
