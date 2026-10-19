"""v7 (PREREG7): steering in a 2-layer transformer. Runs on a Kaggle GPU; resumable per batch.
Stage A: 8 targets x {original split, steered g=1.0} x 12 fresh inits (9800-9811), 200 steps, records step-200 embedding
         spectra and init readouts.
Stage B: inits 9800, 9801 on both splits of every target (32 runs), up to STEPS_B steps, test acc every 250,
         stop 2000 steps after all runs in a batch grok; final key frequencies from the logit spectrum."""
import os, sys, json, time, numpy as np, torch
import gpu_grok as G, twolayer as T
STEPS_B = int(os.environ.get("V7_STEPS_B", "15000"))
S = np.load("splits.npz"); I = json.load(open("splits_info.json"))
TARGETS, BASE = I["targets"], {int(k): v for k, v in I["base"].items()}
INITS = list(range(9800, 9812)); FULL = INITS[:2]
if os.environ.get("V7_SMOKE"):                       # pipeline test only: 2 targets, 4 inits, short stage B
    TARGETS, INITS, STEPS_B = TARGETS[:2], INITS[:4], 300
OUT = G.OUT; (OUT / "A").mkdir(parents=True, exist_ok=True); (OUT / "B").mkdir(parents=True, exist_ok=True)
bs = 16 if G.DEV.type == "cuda" else 4
t0 = time.time()
for k in TARGETS:
    for g in (0.0, 1.0):
        for c0 in range(0, len(INITS), min(bs, len(INITS))):
            f = OUT / "A" / f"k{k}_g{g}_c{c0}.npz"
            if f.exists(): continue
            ch = INITS[c0:c0 + min(bs, len(INITS))]; tr = torch.as_tensor(S[f"k{k}_g{g}"])
            r = T.train_batch2([dict(init=i, split=BASE[k]) for i in ch], 200, record=(200,), custom_train_idx=[tr] * len(ch))
            np.savez(f, inits=np.array(ch), embed200=r["rec"][200]["embed"], **{f"ro_{n}": v for n, v in r["readouts0"].items()})
    print(f"A k{k} done {time.time()-t0:.0f}s", flush=True)
print("A_DONE", flush=True)
jobs = [(k, g, i) for k in TARGETS for g in (1.0, 0.0) for i in FULL]
for j0 in range(0, len(jobs), bs):
    f = OUT / "B" / f"b{j0:02d}.json"
    if f.exists(): continue
    js = jobs[j0:j0 + bs]
    r = T.train_batch2([dict(init=i, split=BASE[k]) for k, g, i in js], STEPS_B, record=(200,), eval_every=250,
                       stop_after_grok=2000, custom_train_idx=[torch.as_tensor(S[f"k{k}_g{g}"]) for k, g, i in js], verbose=True)
    rows = [dict(k=k, level=g, init=i, grok=r["grok"][m], last_step=r["last_step"], **r["final"][m],
                 top3_200=np.argsort(-r["rec"][200]["embed"][m])[:3].tolist()) for m, (k, g, i) in enumerate(js)]
    f.write_text(json.dumps(rows, default=float)); print(f"B {j0} done {time.time()-t0:.0f}s", flush=True)
print("ALLDONE", flush=True)
