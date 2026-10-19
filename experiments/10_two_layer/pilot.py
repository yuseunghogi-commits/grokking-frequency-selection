"""Development pilot (dev seeds, not used in any confirmatory test): does the 2-layer model grok in this setup, and when?"""
import os, sys, time, json, torch
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(H))
os.environ["GROK_OUT"] = os.path.join(H, "results_pilot"); sys.path.insert(0, os.path.join(R, "experiments", "02_kaggle_v1")); sys.path.insert(0, H)
import twolayer as T, gpu_grok as G
torch.set_num_threads(2)
specs = [dict(init=9990, split=9990), dict(init=9991, split=9991)]
t0 = time.time()
r = T.train_batch2(specs, 15000, record=(0, 200), eval_every=500, stop_after_grok=1000, verbose=True)
for row in r["curves"]: print(row["step"], [round(x, 3) for x in row["test_acc"]], flush=True)
out = dict(grok=r["grok"], final=r["final"], last_step=r["last_step"], minutes=(time.time() - t0) / 60,
           curves=r["curves"], top3_200=[sorted(range(56), key=lambda k: -r["rec"][200]["embed"][m][k])[:3] for m in range(2)])
G.OUT.mkdir(parents=True, exist_ok=True); (G.OUT / "pilot.json").write_text(json.dumps(out, default=float))
print("PILOT_DONE", r["grok"], [f["logit_keys"] for f in r["final"]], [f["neuron_keys"] for f in r["final"]], flush=True)
