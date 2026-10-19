"""Training curves + circuit progress measures across checkpoints."""
import json, glob, re, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze import analyze

final = json.load(open("results.json"))
keys = final["key_freqs"]
log = json.load(open("log.json"))
ck = sorted(glob.glob("ckpt/step*.pt"), key=lambda s: int(re.findall(r"\d+", s)[0]))
rows = []
for c in ck:
    r = analyze(c, figs=False, quiet=True, keys=keys)
    rows.append(dict(step=int(re.findall(r"\d+", c)[0]), restricted=r["restricted"][0],
                     excluded=r["excluded"][0], test=r["test_loss"],
                     embed_share=r["embed_key_share"], neuron_share=r["neuron_var_explained_by_keys"]))
    print(rows[-1], flush=True)
json.dump(rows, open("progress.json", "w"), indent=1)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
ACC, INK, MUT, BLU = "#c2410c", "#1f2937", "#9ca3af", "#2563eb"
st = [l["step"] for l in log]
f, axs = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axs[0]
ax.plot(st, [l["train_acc"] for l in log], color=INK, label="train")
ax.plot(st, [l["test_acc"] for l in log], color=ACC, label="test (unseen pairs)")
ax.set_xlabel("training step"); ax.set_ylabel("accuracy"); ax.legend(frameon=False)
ax.set_title("Accuracy", loc="left")
ax = axs[1]
ax.plot(st, [l["train_loss"] for l in log], color=INK, label="train")
ax.plot(st, [l["test_loss"] for l in log], color=ACC, label="test")
ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("loss (log)")
ax2 = ax.twinx(); ax2.plot(st, [l["wnorm"] for l in log], color=MUT, ls="--", label="weight norm²")
ax2.set_ylabel("weight norm²", color=MUT); ax2.spines["top"].set_visible(False)
ax.legend(frameon=False, loc="upper right"); ax.set_title("Loss", loc="left")
f.tight_layout(); f.savefig("figs/grokking.png", dpi=160); plt.close(f)

f, axs = plt.subplots(1, 2, figsize=(11, 3.8))
s = [r["step"] for r in rows]
ax = axs[0]
ax.plot(s, [r["test"] for r in rows], color=ACC, label="test loss (clean)")
ax.plot(s, [r["restricted"] for r in rows], color=BLU, label="restricted: key freqs only")
ax.plot(s, [r["excluded"] for r in rows], color=INK, label="excluded: key freqs removed")
ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("test loss (log)")
ax.legend(frameon=False, fontsize=9); ax.set_title("Causal progress measures", loc="left")
ax = axs[1]
ax.plot(s, [r["embed_share"] for r in rows], color=ACC, label="embedding")
ax.plot(s, [r["neuron_share"] for r in rows], color=BLU, label="MLP neurons")
ax.set_ylim(0, 1); ax.set_xlabel("training step"); ax.set_ylabel(f"share of variance in k={keys}")
ax.legend(frameon=False); ax.set_title("How much of the model is the circuit", loc="left")
f.tight_layout(); f.savefig("figs/progress.png", dpi=160); plt.close(f)
