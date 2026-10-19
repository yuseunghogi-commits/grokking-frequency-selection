"""v3 confirmation run (CPU). Resumable: one npz per (split, chunk)."""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(os.path.dirname(_HERE))
os.environ.setdefault("GROK_OUT", os.path.join(_HERE, "results"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "02_kaggle_v1"))
import numpy as np, torch
import gpu_grok as G
torch.set_num_threads(2)
SPLITS = list(range(5000, 5016)); INITS = list(range(6000, 6016))
out = G.OUT / "runs"; out.mkdir(parents=True, exist_ok=True)
t0 = time.time(); done = 0
for s in SPLITS:
    for c0 in range(0, len(INITS), 4):
        f = out / f"s{s}_c{c0}.npz"
        if f.exists(): continue
        ch = INITS[c0:c0 + 4]
        r = G.train_batch([dict(init=i, split=s) for i in ch], 200, record=(0, 200))
        np.savez(f, inits=np.array(ch), split=s, embed200=r["rec"][200]["embed"], path0=r["rec"][0]["path"])
        done += 1
        print(f"{s} {c0}: {(time.time()-t0)/done:.0f} s/chunk", flush=True)
print("ALLDONE", flush=True)
