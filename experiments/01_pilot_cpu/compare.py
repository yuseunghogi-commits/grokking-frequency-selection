"""Compare WD=1.0 run (./) with WD=0 run (wd0/): curves + Fourier sparsity of the embedding over training."""
import json, glob, re, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze import load, embed_freq_norms, analyze
torch.set_grad_enabled(False)

def top3_share(path):
    fn = embed_freq_norms(load(path)) ** 2
    return (fn.sort(descending=True).values[:3].sum() / fn.sum()).item()

def ckpts(d):
    return sorted(glob.glob(f"{d}/ckpt/step*.pt"), key=lambda s: int(re.findall(r"step(\d+)", s)[0]))

runs = {"weight decay 1.0": ".", "weight decay 0": "wd0"}
res = {}
for name, d in runs.items():
    log = json.load(open(f"{d}/log.json"))
    sp = [(int(re.findall(r"step(\d+)", c)[0]), top3_share(c)) for c in ckpts(d)]
    res[name] = dict(log=log, sparsity=sp)
    print(name, "final test acc", log[-1]["test_acc"], "wnorm", round(log[-1]["wnorm"]), "top3 share", [round(s, 3) for _, s in sp])

# full circuit analysis on the WD=0 final checkpoint (auto-detected frequencies)
last = ckpts("wd0")[-1]
r = analyze(last, figs=False, quiet=True)
summary = {k: r[k] for k in ["key_freqs", "test_acc", "embed_key_share", "neuron_var_explained_by_keys",
                             "neuron_cluster_sizes", "median_single_freq_var_frac", "restricted", "excluded"]}
print(json.dumps(summary, indent=1))
json.dump(dict(summary=summary, sparsity={k: v["sparsity"] for k, v in res.items()}), open("wd0/compare.json", "w"), indent=1)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = {"weight decay 1.0": "#c2410c", "weight decay 0": "#2563eb"}
f, axs = plt.subplots(1, 3, figsize=(14, 3.8))
for name, v in res.items():
    st = [l["step"] for l in v["log"]]
    axs[0].plot(st, [l["test_acc"] for l in v["log"]], color=C[name], label=name)
    axs[1].plot(st, [l["wnorm"] for l in v["log"]], color=C[name], label=name)
    axs[2].plot([s for s, _ in v["sparsity"]], [x for _, x in v["sparsity"]], "o-", color=C[name], label=name, ms=4)
axs[0].set_title("Test accuracy (unseen pairs)", loc="left"); axs[0].set_ylim(0, 1.02)
axs[1].set_title("Weight norm²", loc="left")
axs[2].set_title("Embedding: share in top-3 frequencies", loc="left"); axs[2].set_ylim(0, 1)
axs[2].axhline(3 / 56, color="#9ca3af", ls=":", lw=1); axs[2].text(12000, 3 / 56 - 0.045, "uniform baseline (3/56)", ha="right", color="#6b7280", fontsize=9)
for a in axs: a.set_xlabel("training step")
axs[0].legend(frameon=False)
f.tight_layout(); f.savefig("figs/wd_compare.png", dpi=160); plt.close(f)
