"""Figure for PREREG5 -> report/figures/v6_steering.png"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(H))
r = json.load(open(os.path.join(H, "results", "V5_results.json"))); I = json.load(open(os.path.join(H, "splits_info.json")))
L = I["levels"]; rate = np.array(r["S1"]["rate_by_target_level"])
sw = {}
for s in I["splits"]: sw.setdefault(s["level"], []).append(s["swapped"])
x = [np.mean(sw[g]) / 3830 * 100 for g in L]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for t, row in zip(I["targets"], rate): ax[0].plot(x, row * 100, color="#9aa7b8", lw=1, alpha=.8)
ax[0].plot(x, rate.mean(0) * 100, color="#d9622b", lw=3, marker="o", label="mean over 8 target frequencies")
ax[0].set_xlabel("training pairs swapped (% of 3830)"); ax[0].set_ylabel("runs with target in top-3 at step 200 (%)")
ax[0].set_title("Early: target becomes a leading frequency"); ax[0].legend(frameon=False); ax[0].set_ylim(-3, 103)
runs = r["S3"]["runs"]
for j, lev in enumerate((0.0, 1.0)):
    rs = [q for q in runs if q["level"] == lev]
    hit = sum(q["k"] in q["keys"] for q in rs)
    ax[1].bar(j, hit / len(rs) * 100, color="#d9622b" if lev else "#9aa7b8", width=.55)
    ax[1].text(j, hit / len(rs) * 100 + 2, f"{hit}/{len(rs)}", ha="center", fontsize=12)
ax[1].set_xticks([0, 1], ["original split", "steered split (~12% swapped)"]); ax[1].set_ylim(0, 110)
ax[1].set_ylabel("fully trained runs with target in final circuit (%)"); ax[1].set_title("Final circuit after 10,000 steps")
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Pre-registered v5: choosing which frequency a network learns by editing only its training split")
fig.tight_layout(); fig.savefig(os.path.join(R, "report", "figures", "v6_steering.png"), dpi=110)
