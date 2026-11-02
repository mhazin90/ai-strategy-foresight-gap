import sys
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

settings = [(t, m) for t in [1, 2, 3] for m in [1, 2, 3]]
paragraphs = pd.read_csv("data/02_paragraphs.csv")
doc_length = paragraphs.groupby("doc_id").size()

frames = []
for t, m in settings:
    subprocess.run([sys.executable, "08_closures.py", "--threshold", str(t), "--min-mentions", str(m)], check=True, capture_output=True)
    subprocess.run([sys.executable, "09_network_metrics.py", "--threshold", str(t), "--min-mentions", str(m), "--coder", "A"], check=True, capture_output=True)
    closures = pd.read_csv(f"data/08_closures_T{t}_M{m}.csv")
    closures = closures[closures.coder == "A"]
    network = pd.read_csv(f"data/09_network_docs_A_T{t}_M{m}.csv")[["doc_id", "mediation", "betweenness_F"]]
    merged = closures.merge(network, on="doc_id")
    merged["T"] = t
    merged["M"] = m
    frames.append(merged)

results = pd.concat(frames)
results.to_csv("data/10_sensitivity.csv", index=False)

baseline = results[(results["T"] == 2) & (results["M"] == 2)].set_index("doc_id")
rows = []
for (t, m), group in results.groupby(["T", "M"]):
    group = group.set_index("doc_id")
    shared = baseline.index.intersection(group.index)
    length = doc_length.reindex(group.index)
    means = group[["rho_OF", "rho_FI", "rho_OI"]].mean()
    rows.append({
        "T": t,
        "M": m,
        "docs": len(group),
        "docs_with_F": int((group.n_F > 0).sum()),
        "density": round(group.density.mean(), 3),
        "rho_OF": round(means.rho_OF, 3),
        "rho_FI": round(means.rho_FI, 3),
        "rho_OI": round(means.rho_OI, 3),
        "OI>OF>FI": bool(means.rho_OI > means.rho_OF > means.rho_FI),
        "mediation": round(group.mediation.mean(), 3),
        "rank_vs_base_density": round(spearmanr(baseline.loc[shared, "density"], group.loc[shared, "density"], nan_policy="omit").statistic, 3),
        "rank_vs_base_mediation": round(spearmanr(baseline.loc[shared, "mediation"], group.loc[shared, "mediation"], nan_policy="omit").statistic, 3),
        "length_vs_density": round(spearmanr(length, group.density, nan_policy="omit").statistic, 3),
        "length_vs_rho_OI": round(spearmanr(length, group.rho_OI, nan_policy="omit").statistic, 3),
        "length_vs_mediation": round(spearmanr(length, group.mediation, nan_policy="omit").statistic, 3),
    })

summary = pd.DataFrame(rows)
summary.to_csv("data/10_sensitivity_summary.csv", index=False)
pd.set_option("display.width", 250)
print(summary.to_string(index=False))
