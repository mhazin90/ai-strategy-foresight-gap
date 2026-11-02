import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

T_OI = 2
MIN_MENTIONS = 2

paragraphs = pd.read_csv("data/02_paragraphs.csv")
inventory = pd.read_csv("data/01_inventory.csv").set_index("id")
doc_sizes = paragraphs.groupby("doc_id").size()

coder_a = {}
with open("data/04_coder_A.jsonl") as f:
    for line in f:
        record = json.loads(line)
        if record["status"] == "ok":
            coder_a[record["para_id"]] = record

detector = {}
for tag in ["A", "C"]:
    detector[tag] = {}
    with open(f"data/04c_foresight_{tag}.jsonl") as f:
        for line in f:
            record = json.loads(line)
            if record["status"] == "ok":
                detector[tag][record["para_id"]] = record

candidates = set(detector["A"]) | set(detector["C"])
versions = {}
for para_id in candidates:
    yes_a = detector["A"].get(para_id, {}).get("foresight", False)
    yes_c = detector["C"].get(para_id, {}).get("foresight", False)
    activities = []
    for tag in ["A", "C"]:
        if detector[tag].get(para_id, {}).get("foresight", False):
            activities += detector[tag][para_id]["activities"]
    if yes_a and yes_c:
        versions.setdefault("strict", {})[para_id] = activities
    if yes_a or yes_c:
        versions.setdefault("broad", {})[para_id] = activities

rows = []
for doc_id in doc_sizes.index:
    records = [r for r in coder_a.values() if r["doc_id"] == doc_id]
    objective_counts = {}
    instrument_counts = {}
    for r in records:
        for o in r["objectives"]:
            objective_counts[o] = objective_counts.get(o, 0) + 1
        for i in r["instruments"]:
            instrument_counts[i] = instrument_counts.get(i, 0) + 1
    objectives = {o for o, n in objective_counts.items() if n >= MIN_MENTIONS}
    instruments = {i for i, n in instrument_counts.items() if n >= MIN_MENTIONS}
    oi_links = {(l["source"], l["target"]) for r in records for l in r["links"] if l["type"] == "OI" and l["strength"] >= T_OI and l["source"] in objectives and l["target"] in instruments}
    row = {
        "doc_id": doc_id,
        "generation": int(inventory.loc[doc_id, "generation"]),
        "paragraphs": int(doc_sizes[doc_id]),
        "n_O": len(objectives),
        "n_I": len(instruments),
        "rho_OI": len(oi_links) / (len(objectives) * len(instruments)) if objectives and instruments else np.nan,
    }
    for version in ["strict", "broad"]:
        doc_foresight = {p: acts for p, acts in versions.get(version, {}).items() if p.startswith(doc_id + "_")}
        foresight_objectives = {o["code"] for acts in doc_foresight.values() for a in acts for o in a["objectives"]}
        foresight_instruments = {i["code"] for acts in doc_foresight.values() for a in acts for i in a["instruments"]}
        mediated = {(o, i) for o, i in oi_links if o in foresight_objectives and i in foresight_instruments}
        row[f"F_{version}"] = len(doc_foresight)
        row[f"F_rate_{version}"] = 100 * len(doc_foresight) / doc_sizes[doc_id]
        row[f"rho_OF_{version}"] = len(foresight_objectives & objectives) / len(objectives) if objectives else np.nan
        row[f"rho_FI_{version}"] = len(foresight_instruments & instruments) / len(instruments) if instruments else np.nan
        row[f"mediation_{version}"] = len(mediated) / len(oi_links) if oi_links else np.nan
    rows.append(row)

docs = pd.DataFrame(rows)
docs.to_csv("data/07b_foresight_docs.csv", index=False)

pd.set_option("display.width", 250)
print(docs[["doc_id", "generation", "paragraphs", "rho_OI", "F_strict", "F_broad", "rho_OF_strict", "rho_OF_broad", "rho_FI_strict", "rho_FI_broad", "mediation_strict", "mediation_broad"]].round(3).to_string(index=False))
print()
print("MEANS")
print(docs[["rho_OI", "rho_OF_strict", "rho_OF_broad", "rho_FI_strict", "rho_FI_broad", "mediation_strict", "mediation_broad"]].mean().round(3).to_string())
print()
print("documents with no foresight at all  strict:", int((docs.F_strict == 0).sum()), " broad:", int((docs.F_broad == 0).sum()), " of", len(docs))
print("rank agreement strict vs broad (foresight rate):", round(spearmanr(docs.F_rate_strict, docs.F_rate_broad).statistic, 3))
print("length vs foresight count (broad):", round(spearmanr(docs.paragraphs, docs.F_broad).statistic, 3))
