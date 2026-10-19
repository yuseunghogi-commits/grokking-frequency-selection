"""v4 generalisation runs (PREREG4): addition and multiplication at p=59 and p=97. Resumable."""
import sys, os, json, time, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src")); torch.set_num_threads(2)
import freqsel as fs
from freqsel.tasks import Task
OUT = os.path.join(ROOT, "results", "v4"); os.makedirs(OUT, exist_ok=True)
SPLITS = list(range(8000, 8008)); INITS = list(range(8100, 8112))
for name, p in [("add", 59), ("mul", 59), ("add", 97), ("mul", 97)]:
    T = Task(name, p)
    for s in SPLITS:
        f = f"{OUT}/{name}{p}_s{s}.npz"
        if os.path.exists(f): continue
        E = []
        for c0 in range(0, 12, 6):
            r = fs.train_batch(T, INITS[c0:c0 + 6], [s] * 6, 100, record=(0, 100)); E.append(r[100]["embed"])
        np.savez(f, embed100=np.concatenate(E)); print(name, p, s, flush=True)
print("ALLDONE", flush=True)
