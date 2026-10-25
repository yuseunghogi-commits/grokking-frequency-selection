"""Figure for PREREG7 -> report/figures/v7_two_layer.png"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(H))
r = json.load(open(os.path.join(H, "results", "V7_results.json")))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
bt = np.array(r["T1"]["by_target"]) * 100
for j, (lab, col) in enumerate((("original split", "#9aa7b8"), ("steered split", "#d9622b"))):
    ax[0].bar(j, bt[:, j].mean(), color=col, width=.55); ax[0].text(j, bt[:, j].mean() + 2, f"{bt[:, j].mean():.0f}%", ha="center")
ax[0].set_xticks([0, 1], ["original split", "steered split"]); ax[0].set_ylim(0, 110)
ax[0].set_ylabel("runs with target in step-200 top-3 (%)"); ax[0].set_title("T1 early (96 runs each)")
t3 = r["T3"]
for j, (v, col) in enumerate(((t3["original"], "#9aa7b8"), (t3["steered"], "#d9622b"))):
    ax[1].bar(j, v / 16 * 100, color=col, width=.55); ax[1].text(j, v / 16 * 100 + 2, f"{v}/16", ha="center")
ax[1].set_xticks([0, 1], ["original split", "steered split"]); ax[1].set_ylim(0, 110)
ax[1].set_ylabel("target in final circuit (%)"); ax[1].set_title("T3 final circuit (logit keys)")
m = r["T4"]["mean_rho"]; names = ["E", "OV1", "ALL", "OV2", "U"]
cols = ["#d9622b" if n == "ALL" else "#5b6b80" for n in names]
ax[2].bar(range(5), [m[n] for n in names], color=cols, width=.6)
ax[2].set_xticks(range(5), ["W_E", "E·OV1", "E·(all paths)\n[predicted best]", "E·OV2", "W_U"])
ax[2].set_ylabel("Spearman with init vote"); ax[2].set_title("T4 init readout (failed: all paths ≤ OV1)")
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Pre-registered v7: steering by the training split also works in a 2-layer transformer")
fig.tight_layout(); fig.savefig(os.path.join(R, "report", "figures", "v7_two_layer.png"), dpi=110)
