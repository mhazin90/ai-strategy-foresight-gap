import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PAIRS = [
    ("GBR", "GBR_2021", "GBR_2025"),
    ("USA", "USA_2019", "USA_2025"),
    ("DEU", "DEU_2018", "DEU_2020"),
    ("FRA", "FRA_2018", "FRA_2021"),
    ("EU", "EU_2018", "EU_2021"),
    ("JPN", "JPN_2019", "JPN_2022"),
    ("SGP", "SGP_2019", "SGP_2023"),
    ("BRA", "BRA_2021", "BRA_2024"),
]

docs = pd.read_csv("data/07b_foresight_docs.csv").set_index("doc_id")
paragraphs = pd.read_csv("data/02_paragraphs.csv")
doc_sizes = paragraphs.groupby("doc_id").size()

coder_a = {}
with open("data/04_coder_A.jsonl") as f:
    for line in f:
        record = json.loads(line)
        if record["status"] == "ok":
            coder_a[record["para_id"]] = record

for doc_id in docs.index:
    records = [r for r in coder_a.values() if r["doc_id"] == doc_id]
    n_eth = sum("ETH" in r["objectives"] for r in records)
    n_regi = sum("REGI" in r["instruments"] for r in records)
    n_soft = sum("SOFT" in r["instruments"] for r in records)
    docs.loc[doc_id, "share_ETH"] = 100 * n_eth / doc_sizes[doc_id]
    docs.loc[doc_id, "share_REGI"] = 100 * n_regi / doc_sizes[doc_id]
    docs.loc[doc_id, "share_SOFT"] = 100 * n_soft / doc_sizes[doc_id]
    docs.loc[doc_id, "binding_ratio"] = n_regi / (n_regi + n_soft) if n_regi + n_soft else np.nan

docs.to_csv("data/12_doc_measures.csv")

METRICS = [
    "rho_OI",
    "F_rate_strict", "F_rate_broad",
    "rho_OF_strict", "rho_OF_broad",
    "rho_FI_strict", "rho_FI_broad",
    "mediation_strict", "mediation_broad",
    "share_ETH", "share_REGI", "share_SOFT", "binding_ratio",
]

pair_rows = []
for country, first, second in PAIRS:
    for metric in METRICS:
        before = docs.loc[first, metric]
        after = docs.loc[second, metric]
        pair_rows.append({"country": country, "metric": metric, "gen1": before, "gen2": after, "change": after - before})
pairs = pd.DataFrame(pair_rows)
pairs.to_csv("data/12_pairs.csv", index=False)

summary_rows = []
for metric in METRICS:
    group = pairs[pairs.metric == metric].dropna(subset=["change"])
    changes = group["change"].values
    nonzero = changes[np.abs(changes) > 1e-12]
    try:
        p_value = wilcoxon(nonzero).pvalue if len(nonzero) >= 3 else np.nan
    except Exception:
        p_value = np.nan
    summary_rows.append({
        "metric": metric,
        "n_pairs": len(group),
        "mean_gen1": round(group["gen1"].mean(), 3),
        "mean_gen2": round(group["gen2"].mean(), 3),
        "mean_change": round(group["change"].mean(), 3),
        "up": int((changes > 1e-12).sum()),
        "down": int((changes < -1e-12).sum()),
        "same": int((np.abs(changes) <= 1e-12).sum()),
        "wilcoxon_p": round(float(p_value), 3) if not np.isnan(p_value) else np.nan,
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv("data/12_summary.csv", index=False)

pd.set_option("display.width", 250)
print("PER COUNTRY (gen1 -> gen2)")
wide = pairs.pivot(index="country", columns="metric", values="change")[["rho_OI", "F_broad" if "F_broad" in METRICS else "F_rate_broad", "mediation_broad", "share_ETH", "share_REGI", "share_SOFT", "binding_ratio"]]
print(wide.round(3).to_string())
print()
print("SUMMARY OVER 8 PAIRS")
print(summary.to_string(index=False))
