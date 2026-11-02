import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

os.makedirs("figures/labeled", exist_ok=True)
os.makedirs("figures/clean", exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

C_OI = "#0072B2"
C_OF = "#E69F00"
C_FI = "#009E73"
C_MED = "#CC79A7"
C_GREY = "#BDBDBD"
C_LIGHT = "#E0E0E0"
C_DARK = "#333333"
C_EU = "#D55E00"

docs = pd.read_csv("data/12_doc_measures.csv").set_index("doc_id")
rare = pd.read_csv("data/10b_rarefied.csv")
reliability = pd.read_csv("data/06_reliability.csv")
pairs = pd.read_csv("data/12_pairs.csv")
total_paragraphs = len(pd.read_csv("data/02_paragraphs.csv"))

records = {}
for name, path in [("A", "data/04_coder_A.jsonl"), ("C", "data/06_coder_C.jsonl"), ("DA", "data/04c_foresight_A.jsonl"), ("DC", "data/04c_foresight_C.jsonl")]:
    latest = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                latest[r["para_id"]] = r
    records[name] = {k: v for k, v in latest.items() if v["status"] == "ok"}

strict_ids = {p for p in records["DA"] if records["DA"][p]["foresight"] and records["DC"].get(p, {}).get("foresight", False)}
broad_ids = {p for p in set(records["DA"]) | set(records["DC"]) if records["DA"].get(p, {}).get("foresight", False) or records["DC"].get(p, {}).get("foresight", False)}
shared_detector = sorted(set(records["DA"]) & set(records["DC"]))
kappa_detector = cohen_kappa_score([int(records["DA"][p]["foresight"]) for p in shared_detector], [int(records["DC"][p]["foresight"]) for p in shared_detector])
kappa_means = reliability.groupby(["pair", "section"])["kappa"].mean()

foresight_shares = [
    100 * np.mean([len(r["foresight"]) > 0 for r in records["A"].values()]),
    100 * np.mean([len(r["foresight"]) > 0 for r in records["C"].values()]),
    100 * len(strict_ids) / total_paragraphs,
    100 * len(broad_ids) / total_paragraphs,
]
links_a = [l for r in records["A"].values() for l in r["links"]]
links_c = [l for r in records["C"].values() for l in r["links"]]
acts_a = [a for r in records["DA"].values() for a in r["activities"]]
acts_c = [a for r in records["DC"].values() for a in r["activities"]]
evidence_shares = [
    100 * np.mean([l["evidence_found"] for l in links_a]),
    100 * np.mean([l["evidence_found"] for l in links_c]),
    100 * np.mean([a["evidence_found"] for a in acts_a]),
    100 * np.mean([a["evidence_found"] for a in acts_c]),
]


def finish(ax, clean, letter=None, xlabel=None, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if clean:
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        legend = ax.get_legend()
        if legend:
            legend.remove()
    else:
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if letter:
            ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


for clean in [False, True]:
    folder = "figures/clean" if clean else "figures/labeled"

    fig = plt.figure(figsize=(7.2, 2.6))
    gs = GridSpec(1, 3, figure=fig, left=0.08, right=0.98, bottom=0.25, top=0.88, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    groups = ["objectives", "instruments", "foresight"]
    x = np.arange(4)
    width = 0.36
    ab = [kappa_means.get(("A-B", g), np.nan) for g in groups] + [np.nan]
    ac = [kappa_means.get(("A-C", g), np.nan) for g in groups] + [kappa_detector]
    ax.bar(x - width / 2, ab, width, color=C_GREY, label="LLM A vs embeddings B")
    ax.bar(x + width / 2, ac, width, color=C_OI, label="LLM A vs LLM C")
    ax.axhline(0, color=C_DARK, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Objectives", "Instruments", "Foresight\n(general)", "Foresight\n(detector)"])
    ax.set_ylim(-0.05, 0.7)
    if not clean:
        ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    finish(ax, clean, "a", None, "Cohen's kappa")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x, foresight_shares, 0.6, color=[C_GREY, C_GREY, C_OF, C_OF])
    ax.set_xticks(x)
    ax.set_xticklabels(["LLM A\n(general)", "LLM C\n(general)", "Detector\nstrict", "Detector\nbroad"])
    finish(ax, clean, "b", None, "Paragraphs with foresight (%)")

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(x, evidence_shares, 0.6, color=[C_OI, C_OI, C_OF, C_OF])
    ax.set_xticks(x)
    ax.set_xticklabels(["LLM A\nlinks", "LLM C\nlinks", "Detector A\nactivities", "Detector C\nactivities"])
    ax.set_ylim(0, 100)
    finish(ax, clean, "c", None, "Evidence quotes verified (%)")

    fig.savefig(f"{folder}/fig2_reliability.png", dpi=600)
    plt.close(fig)

    order = docs.sort_values("rho_OI").index.tolist()
    y = np.arange(len(order))
    names = [d.replace("_", " ") for d in order]

    fig = plt.figure(figsize=(7.2, 7.4))
    gs = GridSpec(2, 2, figure=fig, left=0.12, right=0.98, bottom=0.07, top=0.96, wspace=0.08, hspace=0.32, width_ratios=[2.3, 1], height_ratios=[3, 1.25])

    ax_a = fig.add_subplot(gs[0, 0])
    for yi, d in zip(y, order):
        ax_a.plot([docs.loc[d, "rho_OF_strict"], docs.loc[d, "rho_OF_broad"]], [yi + 0.2, yi + 0.2], color=C_OF, lw=1.6, solid_capstyle="round")
        ax_a.plot([docs.loc[d, "rho_FI_strict"], docs.loc[d, "rho_FI_broad"]], [yi - 0.2, yi - 0.2], color=C_FI, lw=1.6, solid_capstyle="round")
    ax_a.scatter(docs.loc[order, "rho_OF_strict"], y + 0.2, s=10, color="white", edgecolor=C_OF, lw=0.9, zorder=3)
    ax_a.scatter(docs.loc[order, "rho_OF_broad"], y + 0.2, s=10, color=C_OF, zorder=3, label="Objective–foresight (strict to broad)")
    ax_a.scatter(docs.loc[order, "rho_FI_strict"], y - 0.2, s=10, color="white", edgecolor=C_FI, lw=0.9, zorder=3)
    ax_a.scatter(docs.loc[order, "rho_FI_broad"], y - 0.2, s=10, color=C_FI, zorder=3, label="Foresight–instrument (strict to broad)")
    ax_a.scatter(docs.loc[order, "rho_OI"], y, s=22, marker="D", color=C_OI, zorder=4, label="Objective–instrument")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(names, fontsize=6.5)
    ax_a.set_xlim(-0.03, 1.03)
    ax_a.set_ylim(-0.7, len(order) - 0.3)
    ax_a.grid(axis="x", color=C_LIGHT, lw=0.5)
    ax_a.set_axisbelow(True)
    if not clean:
        ax_a.legend(frameon=False, fontsize=6.5, loc="lower right")
    finish(ax_a, clean, "a", "Link density", None)

    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_b.barh(y, docs.loc[order, "F_rate_broad"], 0.6, color=C_OF, alpha=0.35, label="Broad")
    ax_b.barh(y, docs.loc[order, "F_rate_strict"], 0.6, color=C_OF, label="Strict")
    ax_b.grid(axis="x", color=C_LIGHT, lw=0.5)
    ax_b.set_axisbelow(True)
    plt.setp(ax_b.get_yticklabels(), visible=False)
    if not clean:
        ax_b.legend(frameon=False, fontsize=6.5, loc="lower right")
    finish(ax_b, clean, "b", "Foresight paragraphs per 100", None)
    if not clean:
        ax_b.tick_params(labelleft=False)

    ax_c = fig.add_subplot(gs[1, :])
    conditions = [("Full length", docs), ("Rarefied N=120", rare[rare.N == 120]), ("Rarefied N=47", rare[rare.N == 47])]
    measures = [("rho_OI", None, C_OI, "Objective–instrument"), ("rho_OF_broad", "rho_OF_strict", C_OF, "Objective–foresight"), ("rho_FI_broad", "rho_FI_strict", C_FI, "Foresight–instrument"), ("mediation_broad", "mediation_strict", C_MED, "Mediation")]
    width = 0.2
    for ci, (cname, data) in enumerate(conditions):
        for mi, (broad, strict, color, label) in enumerate(measures):
            xpos = ci + (mi - 1.5) * width
            ax_c.bar(xpos, data[broad].mean(), width * 0.9, color=color, label=label if ci == 0 else None)
            if strict:
                ax_c.hlines(data[strict].mean(), xpos - width * 0.45, xpos + width * 0.45, color=C_DARK, lw=1.4)
    ax_c.set_xticks(range(3))
    ax_c.set_xticklabels([c[0] for c in conditions])
    ax_c.set_ylim(0, 0.7)
    if not clean:
        ax_c.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper right")
    finish(ax_c, clean, "c", None, "Mean density")

    fig.savefig(f"{folder}/fig3_main_finding.png", dpi=600)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 2, figure=fig, left=0.09, right=0.93, bottom=0.08, top=0.95, wspace=0.45, hspace=0.38)
    slope_metrics = [("rho_OF_broad", "Objective–foresight density", "a"), ("rho_FI_broad", "Foresight–instrument density", "b"), ("mediation_broad", "Foresight mediation", "c"), ("share_ETH", "Ethics paragraphs (%)", "d")]
    for k, (metric, ylabel, letter) in enumerate(slope_metrics):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        sub = pairs[pairs.metric == metric]
        for _, row in sub.iterrows():
            is_eu = row["country"] == "EU"
            ax.plot([0, 1], [row["gen1"], row["gen2"]], color=C_EU if is_eu else C_GREY, lw=1.8 if is_eu else 1.0, marker="o", ms=3.5, zorder=3 if is_eu else 2)
            if not clean:
                ax.text(1.06, row["gen2"], row["country"], fontsize=6, va="center", color=C_EU if is_eu else "#777777")
        ax.plot([0, 1], [sub["gen1"].mean(), sub["gen2"].mean()], color=C_DARK, lw=2.4, marker="s", ms=4.5, zorder=4)
        ax.set_xlim(-0.2, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["1st generation", "2nd generation"])
        finish(ax, clean, letter, None, ylabel)
    fig.savefig(f"{folder}/fig4_generations.png", dpi=600)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 2.9))
    gs = GridSpec(1, 2, figure=fig, left=0.08, right=0.98, bottom=0.2, top=0.88, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    bias_metrics = [("rho_OI", "Obj–inst"), ("rho_OF_broad", "Obj–fore"), ("rho_FI_broad", "Fore–inst"), ("mediation_broad", "Mediation"), ("F_rate_broad", "Foresight rate")]
    r120 = rare[rare.N == 120].set_index("doc_id")
    r47 = rare[rare.N == 47].set_index("doc_id")
    x = np.arange(len(bias_metrics))
    width = 0.26
    values_full = [spearmanr(docs["paragraphs"], docs[m]).statistic for m, _ in bias_metrics]
    values_120 = [spearmanr(docs.loc[r120.index, "paragraphs"], r120[m]).statistic for m, _ in bias_metrics]
    values_47 = [spearmanr(docs.loc[r47.index, "paragraphs"], r47[m]).statistic for m, _ in bias_metrics]
    ax.bar(x - width, values_full, width, color=C_DARK, label="Full length")
    ax.bar(x, values_120, width, color="#7F7F7F", label="Rarefied N=120")
    ax.bar(x + width, values_47, width, color=C_GREY, label="Rarefied N=47")
    ax.axhline(0, color=C_DARK, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in bias_metrics])
    ax.set_ylim(-0.7, 0.8)
    if not clean:
        ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=3)
    finish(ax, clean, "a", None, "Correlation with document length")

    ax = fig.add_subplot(gs[0, 1])
    xs = [0, 1, 2]
    series = [("rho_OI", C_OI, "-", "Objective–instrument"), ("rho_OF_broad", C_OF, "-", "Objective–foresight broad"), ("rho_OF_strict", C_OF, "--", "Objective–foresight strict"), ("rho_FI_broad", C_FI, "-", "Foresight–instrument broad"), ("rho_FI_strict", C_FI, "--", "Foresight–instrument strict"), ("mediation_broad", C_MED, "-", "Mediation broad"), ("mediation_strict", C_MED, "--", "Mediation strict")]
    for metric, color, style, label in series:
        values = [docs[metric].mean(), rare[rare.N == 120][metric].mean(), rare[rare.N == 47][metric].mean()]
        ax.plot(xs, values, color=color, ls=style, lw=1.5, marker="o", ms=3.5, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Full length", "N=120", "N=47"])
    ax.set_ylim(0, 0.7)
    if not clean:
        ax.legend(frameon=False, fontsize=6, loc="upper right", ncol=2)
    finish(ax, clean, "b", None, "Mean density")
    fig.savefig(f"{folder}/fig5_robustness.png", dpi=600)
    plt.close(fig)

print("FIGURE VALUES")
print("kappa A-B:", {g: round(kappa_means.get(("A-B", g), np.nan), 3) for g in ["objectives", "instruments", "foresight"]})
print("kappa A-C:", {g: round(kappa_means.get(("A-C", g), np.nan), 3) for g in ["objectives", "instruments", "foresight"]}, " detector:", round(kappa_detector, 3))
print("foresight shares (%): A general, C general, strict, broad:", [round(v, 2) for v in foresight_shares])
print("evidence verified (%): A links, C links, detector A, detector C:", [round(v, 1) for v in evidence_shares])
print("document order in fig 3 (bottom to top):", ", ".join(order))
print("saved:", sorted(os.listdir("figures/labeled")))
