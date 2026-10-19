"""Factorial init x split runs at 500 steps, resumable. usage: python3 factorial.py <name> <init0> <split0> <n>
Writes fact/<name>/i{init}_s{split}.npz with spectra at recorded steps."""
import sys, os, time, itertools, numpy as np, torch
from batched import train_batch
torch.set_num_threads(2)
name, i0, s0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
out = f"fact/{name}"; os.makedirs(out, exist_ok=True)
REC = (0, 20, 50, 100, 200, 300, 500)
cells = [(i, s) for s in range(s0, s0 + n) for i in range(i0, i0 + n)]
todo = [c for c in cells if not os.path.exists(f"{out}/i{c[0]}_s{c[1]}.npz")]
print(f"{len(todo)} of {len(cells)} cells to run", flush=True)
BATCH = 4
t0 = time.time()
for j in range(0, len(todo), BATCH):
    chunk = todo[j:j + BATCH]
    rec = train_batch([c[0] for c in chunk], [c[1] for c in chunk], steps=500, record=REC)
    for m, (i, s) in enumerate(chunk):
        tmp = f"{out}/i{i}_s{s}.tmp.npz"
        np.savez(tmp, steps=np.array(REC),
                 embed=np.stack([rec[t]["embed"][m] for t in REC]),
                 attn=np.stack([rec[t]["attn"][m] for t in REC]),
                 train_acc=np.array([rec[t]["train_acc"][m] for t in REC]))
        os.replace(tmp, f"{out}/i{i}_s{s}.npz")
    done = j + len(chunk)
    el = time.time() - t0
    print(f"DONE {done}/{len(todo)}  {el/60:.1f} min  ETA {el/done*(len(todo)-done)/60:.0f} min", flush=True)
print("ALLDONE", flush=True)
