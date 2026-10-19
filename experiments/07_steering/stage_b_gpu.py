"""v5 stage B on a GPU (Kaggle). Runs the stage-B jobs that are not already in results/B (pass their indices out),
in one batch, exactly as run5.py (10000 steps, eval every 250, same splits and inits). Writes OUT/B_gpu.json."""
import os, sys, json, time, numpy as np, torch
import gpu_grok as G
S = np.load("splits.npz"); I = json.load(open("splits_info.json"))
TARGETS, LEVELS, BASE = I["targets"], I["levels"], {int(k): v for k, v in I["base"].items()}
FULL_INITS = [9600, 9601]
jobs = [(k, g, i) for k in TARGETS for g in (LEVELS[-1], 0.0) for i in FULL_INITS]
done = int(os.environ.get("V5_SKIP", "4"))          # first 4 jobs (b00.json) were run on CPU
js = jobs[done:]
print("device", G.DEV, "| jobs", len(js), flush=True); t0 = time.time()
out = []
bs = 16 if G.DEV.type == "cuda" else 4
for j0 in range(0, len(js), bs):
    part = js[j0:j0 + bs]
    r = G.train_batch([dict(init=i, split=BASE[k]) for k, g, i in part], 10000, record=(200,), eval_every=250,
                      custom_train_idx=[torch.as_tensor(S[f"k{k}_g{g}"]) for k, g, i in part])
    out += [dict(k=k, level=g, init=i, grok=r["grok"][m], final=r["final"][m],
                 top3_200=np.argsort(-r["rec"][200]["embed"][m])[:3].tolist()) for m, (k, g, i) in enumerate(part)]
    (G.OUT / "B_gpu.json").write_text(json.dumps(out, default=float))
    print(f"batch {j0 // bs + 1} done, {time.time() - t0:.0f}s", flush=True)
print("ALLDONE", flush=True)
