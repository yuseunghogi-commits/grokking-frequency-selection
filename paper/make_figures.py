"""Regenerate every figure of the paper from the stored results (no training). python paper/make_figures.py"""
import os, sys, json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(R, "paper", "figures"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(R, "experiments", "07_steering")); from steer import D_of, split_perm, P
J = lambda *p: json.load(open(os.path.join(R, *p)))
# palette (validated: dataviz/scripts/validate_palette.js on #ffffff) + ink
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, BASE, LIGHT = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#d9d8d2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight",
                     "lines.linewidth": 2, "figure.dpi": 150})
def save(fig, name): fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=200); plt.close(fig)
def panel(ax, s): pass
def T(ax, letter, text): ax.set_title(f"({letter}) {text}", loc="left", fontsize=8.8)
def barlabels(ax, bars, fmt="{:.0f}", dy=0.01, color=INK):
    top = ax.get_ylim()[1]
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy * top, fmt.format(b.get_height()), ha="center", va="bottom", fontsize=8, color=color)
def zlog(x): l = np.log(x); return (l - l.mean(-1, keepdims=True)) / l.std(-1, keepdims=True)

# ---------------- Fig 1: the phenomenon (pilot) ----------------
log = J("experiments", "01_pilot_cpu", "log.json"); res = J("experiments", "01_pilot_cpu", "results.json")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
st = [x["step"] for x in log]
ax[0].plot(st, [x["train_acc"] * 100 for x in log], color=INK2, lw=1.6, label="train")
ax[0].plot(st, [x["test_acc"] * 100 for x in log], color=BLUE, label="test")
ax[0].set_xlabel("training step"); ax[0].set_ylabel("accuracy (%)"); T(ax[0], "a", "Grokking on (a + b) mod 113")
ax[0].legend(loc="center right"); panel(ax[0], "a")
en = np.array(res["embed_freq_norms"]); keys = res["key_freqs"]; k = np.arange(1, 57)
ax[1].bar(k, en / en.sum() * 100, color=[BLUE if i in keys else LIGHT for i in k], width=.8)
for kk in keys: ax[1].text(kk, en[kk - 1] / en.sum() * 100 + 1, f"k={kk}", ha="center", fontsize=8, color=INK)
ax[1].set_xlabel("frequency k"); ax[1].set_ylabel("share of embedding power (%)"); T(ax[1], "b", "After grokking: three key frequencies")
ax[1].grid(axis="x", visible=False); panel(ax[1], "b")
fig.tight_layout(); save(fig, "fig1_phenomenon")

# ---------------- Fig 2: shared answers and D(k) ----------------
tr = split_perm(5000)[0]; a, b = tr // P, tr % P; c = (a + b) % P
N1 = np.zeros((P, P)); np.add.at(N1, (a, c), 1); N = N1 @ N1.T
Dsp = D_of(tr); I3 = list(range(6000, 6016)); S3 = list(range(5000, 5016)); E3 = np.full((16, 16, 56), np.nan)
for f in glob.glob(os.path.join(R, "experiments", "04_cpu_v3_formula", "results", "runs", "*.npz")):
    d = np.load(f); s = S3.index(int(d["split"]))
    for j, i in enumerate(d["inits"]): E3[I3.index(int(i)), s] = d["embed200"][j]
Z3 = zlog(E3); V3 = Z3.mean(0); V3 = V3 - V3.mean(0, keepdims=True)
D3 = np.stack([np.log(D_of(split_perm(s)[0])) for s in S3]); D3c = D3 - D3.mean(0, keepdims=True)
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.45), gridspec_kw=dict(width_ratios=[1, 1.25, 1]))
im = ax[0].imshow(N - np.diag(np.diag(N)), cmap="Blues", interpolation="nearest")
T(ax[0], "a", "Shared answers N[a, a′]"); ax[0].set_xlabel("a′"); ax[0].set_ylabel("a"); ax[0].grid(False)
cb = fig.colorbar(im, ax=ax[0], fraction=.046, pad=.04); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7); panel(ax[0], "a")
v = V3[0]; dd = D3c[0]
ax[1].bar(k, (dd - dd.mean()) / dd.std(), color=LIGHT, width=.8, label="D(k), from the split only")
ax[1].plot(k, (v - v.mean()) / v.std(), color=BLUE, lw=1.4, marker="o", ms=2.5, label="data vote, 16 trained models")
ax[1].set_xlabel("frequency k"); ax[1].set_ylabel("z-score (split-specific part)")
T(ax[1], "b", f"One unseen split (ρ = {spearmanr(dd, v).correlation:.2f})"); ax[1].legend(loc="upper left", fontsize=6.8, ncol=1); ax[1].set_ylim(-3.2, 4.4); ax[1].grid(axis="x", visible=False); panel(ax[1], "b")
for i in range(16): ax[2].scatter(rankdata(D3c[i]) / 56, rankdata(V3[i]) / 56, s=5, color=BLUE, alpha=.35, lw=0)
ax[2].plot([0, 1], [0, 1], color=BASE, lw=1, ls="--")
ax[2].set_xlabel("rank of D(k) within split"); ax[2].set_ylabel("rank of data vote"); T(ax[2], "c", "16 splits, v3 (ρ = 0.71)")
ax[2].set_aspect("equal"); panel(ax[2], "c")
fig.tight_layout(); save(fig, "fig2_dk")

# ---------------- Fig 3: init x split factorial and the anatomy of the data vote ----------------
S1 = J("experiments", "02_kaggle_v1", "results", "SUMMARY.json")["F"]["C1"]; TF = J("experiments", "04_cpu_v3_formula", "results", "training_free_v1.json")
IS = J("experiments", "08_init_theory", "results", "init_side.json"); V2 = J("experiments", "03_kaggle_v2", "results", "V2_results.json")
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1.15]))
labs = ["chance", "init\nvote", "data\nvote", "init +\ndata"]; vals = [S1["chance"], S1["hits"]["init"], S1["hits"]["data"], S1["hits"]["both"]]
bb = ax[0].bar(range(4), vals, color=[LIGHT, INK2, ORANGE, BLUE], width=.62); ax[0].set_ylim(0, 1450); barlabels(ax[0], bb)
ax[0].set_xticks(range(4), labs); ax[0].set_ylabel("top-3 hits (of 3072)"); T(ax[0], "a", "Held-out 32 × 32 factorial"); ax[0].grid(axis="x", visible=False); panel(ax[0], "a")
conds = [("a→a+t", "H2"), ("MLP\nonly", "H4"), ("shuffled", "H1"), ("a−b", "H3")]
vals = [V2[h]["agreement"] for _, h in conds]; ceil = np.mean([V2[h]["ceiling"] for _, h in conds])
bb = ax[1].bar(range(4), vals, color=[BLUE, BLUE, LIGHT, LIGHT], width=.62); ax[1].axhline(ceil, color=INK2, lw=1, ls="--")
ax[1].text(3.4, ceil + .02, "noise ceiling", ha="right", fontsize=7.5, color=INK2); ax[1].set_ylim(0, 1.1); barlabels(ax[1], bb, "{:.2f}")
ax[1].set_xticks(range(4), [c for c, _ in conds], fontsize=7.5); ax[1].set_ylabel("agreement with data vote"); T(ax[1], "b", "What casts the data vote"); ax[1].grid(axis="x", visible=False); panel(ax[1], "b")
tc = V2["timecourse"]
for key, lab, col in (("embed", "embedding", BLUE), ("path", "OV path", AQUA), ("unembed", "unembedding", ORANGE)):
    s_, y_ = np.array(tc[key]["steps"]), np.array(tc[key]["agreement_with_true_vote"]); m = s_ <= 200
    ax[2].plot(s_[m], y_[m], color=col, label=lab, lw=1.6)
ax[2].set_xlabel("training step"); ax[2].set_ylabel("agreement with final vote"); T(ax[2], "c", "It forms within ~20 steps"); ax[2].legend(fontsize=7.5, loc="lower right"); panel(ax[2], "c")
fig.tight_layout(); save(fig, "fig3_votes")

# ---------------- Fig 4: generalisation of D(k) ----------------
V4 = J("results", "v4", "V4_results.json"); V7 = J("experiments", "10_two_layer", "results", "V7_results.json")
frac = {0.15: 0.651, 0.3: 0.728, 0.5: 0.641, 0.7: 0.648}                       # exploratory sweep (frac_an.py output)
rows = [("a+b mod 113 (v3)", 0.709, True), ("a+b mod 59", V4["add59"]["spearman"], True), ("a+b mod 97", V4["add97"]["spearman"], True),
        ("a·b mod 59", V4["mul59"]["spearman"], True), ("a·b mod 97", V4["mul97"]["spearman"], True),
        ("2-layer model (v7)", V7["T2"]["spearman"], True)] + [(f"train fraction {f}", r, False) for f, r in frac.items()]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
y = np.arange(len(rows))[::-1]
bb = ax[0].barh(y, [r[1] for r in rows], color=[BLUE if r[2] else "white" for r in rows], edgecolor=[BLUE] * len(rows), height=.62, lw=1.2)
for yi, r in zip(y, rows): ax[0].text(r[1] + .01, yi, f"{r[1]:.2f}", va="center", fontsize=7.5, color=INK)
ax[0].axvline(0.3, color=INK2, lw=1, ls="--")
ax[0].set_yticks(y, [r[0] for r in rows], fontsize=7.5); ax[0].set_xlim(0, 1.3); ax[0].set_xticks([0, .2, .4, .6, .8, 1.0]); ax[0].set_ylim(-1.1, len(rows) - .4); ax[0].set_xlabel("within-split Spearman(D, data vote)")
T(ax[0], "a", "D(k) vs data vote, all settings"); ax[0].grid(axis="y", visible=False); panel(ax[0], "a")
ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, fc="white", ec=BLUE), plt.Line2D([], [], color=INK2, ls="--", lw=1)],
             labels=["pre-registered", "exploratory", "threshold 0.3"], loc="lower right", fontsize=7)
pts = [(V4[t]["theory"], V4[t]["D_relative_spread"], lab) for t, lab in (("add59", "+59"), ("add97", "+97"), ("mul59", "×59"), ("mul97", "×97"))]
pts += [(1 / np.sqrt(113), s, f"f={f}") for f, s in ((0.15, .095), (0.3, .096), (0.5, .099), (0.7, .096))]
ax[1].plot([.08, .14], [.08, .14], color=BASE, ls="--", lw=1)
ax[1].fill_between([.08, .14], [.06, .105], [.10, .175], color=GRID, alpha=.6, lw=0, label="±25 % (pre-registered)")
for i, (x, yv, lab) in enumerate(pts):
    ax[1].scatter(x, yv, color=BLUE if i < 4 else AQUA, s=26, zorder=3, edgecolor="white", lw=.8)
ax[1].annotate("p = 59", (pts[0][0], pts[0][1]), (0.122, 0.094), fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=BASE, lw=.7))
ax[1].annotate("p = 97", (pts[1][0], pts[1][1]), (0.103, 0.083), fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=BASE, lw=.7))
ax[1].annotate("p = 113,\n4 fractions", (1 / np.sqrt(113), .096), (0.084, 0.118), fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=BASE, lw=.7))
ax[1].set_xlim(.08, .14); ax[1].set_ylim(.07, .15); ax[1].set_xlabel("predicted spread 1/√n"); ax[1].set_ylabel("observed spread of D (std/mean)")
T(ax[1], "b", "Spread of D across k"); ax[1].legend(fontsize=7, loc="upper left"); panel(ax[1], "b")
fig.tight_layout(); save(fig, "fig4_generalisation")

# ---------------- Fig 5: steering ----------------
V5 = J("experiments", "07_steering", "results", "V5_results.json"); SI = J("experiments", "07_steering", "splits_info.json")
L = SI["levels"]; rate = np.array(V5["S1"]["rate_by_target_level"]) * 100
sw = {}; [sw.setdefault(s["level"], []).append(s["swapped"]) for s in SI["splits"]]; x = [np.mean(sw[g]) / 3830 * 100 for g in L]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.75), gridspec_kw=dict(width_ratios=[1, 1.15]))
for r_ in rate: ax[0].plot(x, r_, color=LIGHT, lw=1)
ax[0].plot(x, rate.mean(0), color=BLUE, marker="o", ms=5, label="mean of 8 random targets")
ax[0].plot([], [], color=LIGHT, lw=1, label="individual targets"); ax[0].set_ylim(-3, 104)
ax[0].set_xlabel("training pairs swapped (% of 3830)"); ax[0].set_ylabel("target in step-200 top-3 (%)")
T(ax[0], "a", "Dose–response (1-layer, 96 runs/point)"); ax[0].legend(fontsize=7.5, loc="lower right"); panel(ax[0], "a")
groups = [("1-layer\nstep 200", rate[:, 0].mean(), rate[:, -1].mean()), ("1-layer\nfinal circuit", V5["S3"]["base_key"] / 16 * 100, V5["S3"]["steered_key"] / 16 * 100),
          ("2-layer\nstep 200", V7["T1"]["rate_original"] * 100, V7["T1"]["rate_steered"] * 100), ("2-layer\nfinal circuit", V7["T3"]["original"] / 16 * 100, V7["T3"]["steered"] / 16 * 100)]
xs = np.arange(4); w = .36
b0 = ax[1].bar(xs - w / 2 - .01, [g[1] for g in groups], w, color=LIGHT, label="original split")
b1 = ax[1].bar(xs + w / 2 + .01, [g[2] for g in groups], w, color=BLUE, label="steered split (~12 % swapped)")
ax[1].set_ylim(0, 138); ax[1].set_yticks([0, 20, 40, 60, 80, 100]); barlabels(ax[1], b0, "{:.0f}", color=INK2); barlabels(ax[1], b1, "{:.0f}")
ax[1].set_xticks(xs, [g[0] for g in groups], fontsize=7.5); ax[1].set_ylabel("runs with target selected (%)")
T(ax[1], "b", "Steering, early and in the final circuit"); ax[1].legend(fontsize=7.5, loc="upper left", ncol=2); ax[1].grid(axis="x", visible=False); panel(ax[1], "b")
fig.tight_layout(); save(fig, "fig5_steering")

# ---------------- Fig 6: init side ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
names = [("E", "W_E"), ("EOV", "W_E·OV"), ("U", "W_U")]; sets = [("B_dev", "v1 dev"), ("B_test", "v1 test"), ("v3", "v3")]
cols = [INK2, BLUE, AQUA]; w = .22
for j, (key, lab) in enumerate(names + [("prod", "W_E·OV × W_U\n(product rule)")]):
    vals = [IS[s]["T1"][key]["rho_init_vote"] if key != "prod" else IS[s]["T2"]["1.0"]["rho_init_vote"] for s, _ in sets]
    ax[0].bar(np.arange(3) + (j - 1.5) * w, vals, w * .92, color=[INK2, BLUE, LIGHT, ORANGE][j], label=lab.replace("\n", " "))
ax[0].set_xticks(range(3), [l for _, l in sets]); ax[0].set_ylim(0, 1.12); ax[0].set_ylabel("Spearman with init vote")
T(ax[0], "a", "1-layer init readouts"); ax[0].legend(fontsize=7, ncol=2, loc="upper left"); ax[0].grid(axis="x", visible=False); panel(ax[0], "a")
m = V7["T4"]["mean_rho"]; order = [("E", "W_E"), ("OV1", "E·OV₁"), ("ALL", "E·(all paths)\npredicted best"), ("OV2", "E·OV₂"), ("U", "W_U")]
bb = ax[1].bar(range(5), [m[k_] for k_, _ in order], color=[INK2, BLUE, ORANGE, LIGHT, LIGHT], width=.62); barlabels(ax[1], bb, "{:.2f}")
ax[1].set_xticks(range(5), [l for _, l in order], fontsize=7); ax[1].set_ylim(0, 0.9); ax[1].set_ylabel("Spearman with init vote")
T(ax[1], "b", "2-layer (pre-registered T4 fails)"); ax[1].grid(axis="x", visible=False); panel(ax[1], "b")
fig.tight_layout(); save(fig, "fig6_init")

# ---------------- Fig 7: the training-fraction law (failed) ----------------
V6 = J("experiments", "09_fraction_law", "results", "V6_results.json")
obs = {0.15: 1.084, 0.3: 1.0, 0.5: 0.896, 0.7: 0.567}                          # exploratory, docs/theory_init_side.md
f_ = np.linspace(0.1, 0.9, 100)
fig, ax = plt.subplots(figsize=(3.6, 2.6))
ax.plot(f_, np.sqrt((1 - f_) / 0.7), color=BLUE, lw=1.6, label="√(1 − f) law (derived post hoc)")
ax.plot(f_, (1 - f_) / 0.7, color=INK2, lw=1.2, ls="--", label="(1 − f) law")
ax.scatter(list(obs), list(obs.values()), color=AQUA, s=28, zorder=3, edgecolor="white", label="sweep used to derive the law")
ax.vlines(0.85, *V6["R_boot_ci95"], color=ORANGE, lw=1.5, zorder=4)
ax.scatter([0.85], [V6["R"]], color=ORANGE, s=40, zorder=5, edgecolor="white", label="pre-registered test (f = 0.85)")
ax.fill_between([0.83, 0.87], [0.7 * V6["prediction_sqrt_law"]] * 2, [1.3 * V6["prediction_sqrt_law"]] * 2, color=BLUE, alpha=.15, lw=0)
ax.set_xlabel("training fraction f"); ax.set_ylabel("data / init amplitude\n(relative to f = 0.3)"); ax.set_ylim(0, 1.25)
ax.set_title("A failed prediction"); ax.legend(fontsize=6.5, loc="lower left")
fig.tight_layout(); save(fig, "fig7_fraction")
print("figures written to", OUT)
