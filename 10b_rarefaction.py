import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

R = 500
T_OI = 2
MIN_MENTIONS = 2
SIZES = [47, 120]
PAIRS = [("GBR", "GBR_2021", "GBR_2025"), ("USA", "USA_2019", "USA_2025"), ("DEU", "DEU_2018", "DEU_2020"), ("FRA", "FRA_2018", "FRA_2021"), ("EU", "EU_2018", "EU_2021"), ("JPN", "JPN_2019", "JPN_2022"), ("SGP", "SGP_2019", "SGP_2023"), ("BRA", "BRA_2021", "BRA_2024")]

paragraphs = pd.read_csv("data/02_paragraphs.csv")
doc_ids = sorted(paragraphs.doc_id.unique())
doc_length = paragraphs.groupby("doc_id").size()
doc_paragraphs = {d: list(paragraphs[paragraphs.doc_id == d].para_id) for d in doc_ids}
measures = pd.read_csv("data/12_doc_measures.csv").set_index("doc_id")
original_clusters = pd.read_csv("data/11_clusters.csv").set_index("doc_id")["cluster"]

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
versions = {"strict": {}, "broad": {}}
for para_id in set(detector["A"]) | set(detector["C"]):
    yes_a = detector["A"].get(para_id, {}).get("foresight", False)
    yes_c = detector["C"].get(para_id, {}).get("foresight", False)
    activities = []
    for tag in ["A", "C"]:
        if detector[tag].get(para_id, {}).get("foresight", False):
            activities += detector[tag][para_id]["activities"]
    if yes_a and yes_c:
        versions["strict"][para_id] = activities
    if yes_a or yes_c:
        versions["broad"][para_id] = activities

info = {}
for para_id in paragraphs.para_id:
    record = coder_a.get(para_id, {"objectives": [], "instruments": [], "links": []})
    info[para_id] = {
        "obj": set(record["objectives"]),
        "ins": set(record["instruments"]),
        "oi": {(l["source"], l["target"]) for l in record["links"] if l["type"] == "OI" and l["strength"] >= T_OI},
    }

rng = np.random.default_rng(7)
rows = []
for N in SIZES:
    for doc_id in doc_ids:
        ids = doc_paragraphs[doc_id]
        if len(ids) < N:
            continue
        values = {}
        for r in range(R):
            sample = rng.choice(ids, N, replace=False)
            obj_count = {}
            ins_count = {}
            oi_all = set()
            for p in sample:
                for o in info[p]["obj"]:
                    obj_count[o] = obj_count.get(o, 0) + 1
                for i in info[p]["ins"]:
                    ins_count[i] = ins_count.get(i, 0) + 1
                oi_all |= info[p]["oi"]
            objectives = {o for o, n in obj_count.items() if n >= MIN_MENTIONS}
            instruments = {i for i, n in ins_count.items() if n >= MIN_MENTIONS}
            oi_links = {(o, i) for o, i in oi_all if o in objectives and i in instruments}
            values.setdefault("rho_OI", []).append(len(oi_links) / (len(objectives) * len(instruments)) if objectives and instruments else np.nan)
            for version in ["strict", "broad"]:
                acts = [versions[version][p] for p in sample if p in versions[version]]
                f_obj = {o["code"] for a_list in acts for a in a_list for o in a["objectives"]}
                f_ins = {i["code"] for a_list in acts for a in a_list for i in a["instruments"]}
                mediated = {(o, i) for o, i in oi_links if o in f_obj and i in f_ins}
                values.setdefault(f"F_rate_{version}", []).append(100 * len(acts) / N)
                values.setdefault(f"rho_OF_{version}", []).append(len(f_obj & objectives) / len(objectives) if objectives else np.nan)
                values.setdefault(f"rho_FI_{version}", []).append(len(f_ins & instruments) / len(instruments) if instruments else np.nan)
                values.setdefault(f"mediation_{version}", []).append(len(mediated) / len(oi_links) if oi_links else np.nan)
        row = {"N": N, "doc_id": doc_id}
        for key, v in values.items():
            row[key] = float(np.nanmean(v)) if not np.all(np.isnan(v)) else 0.0
        rows.append(row)

rare = pd.DataFrame(rows)
rare.to_csv("data/10b_rarefied.csv", index=False)

pd.set_option("display.width", 250)
check_metrics = ["rho_OI", "F_rate_broad", "rho_OF_broad", "rho_FI_broad", "mediation_broad"]
for N in SIZES:
    sub = rare[rare.N == N].set_index("doc_id")
    docs_here = list(sub.index)
    print("=" * 70)
    print(f"N = {N} paragraphs per document, {len(docs_here)} documents")

    print("LENGTH BIAS (Spearman with document length): original -> rarefied")
    for m in check_metrics:
        before = spearmanr(doc_length[docs_here], measures.loc[docs_here, m]).statistic
        after = spearmanr(doc_length[docs_here], sub[m]).statistic
        print(f"   {m:16s} {before:6.3f} -> {after:6.3f}")

    means = sub[["rho_OI", "rho_OF_strict", "rho_OF_broad", "rho_FI_strict", "rho_FI_broad", "mediation_strict", "mediation_broad"]].mean().round(3)
    print("MEANS:", means.to_dict())

    feat = pd.DataFrame(index=docs_here)
    for name in ["F_rate", "rho_OF", "rho_FI", "mediation"]:
        feat[name] = (sub[f"{name}_strict"] + sub[f"{name}_broad"]) / 2
    feat["rho_OI"] = sub["rho_OI"]
    feat["share_ETH"] = measures.loc[docs_here, "share_ETH"]
    feat["binding_ratio"] = measures.loc[docs_here, "binding_ratio"].fillna(measures["binding_ratio"].mean())
    X = feat.values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X)
    ari = adjusted_rand_score(original_clusters[docs_here], labels)
    print("ARCHETYPES: adjusted Rand index vs original clusters:", round(ari, 3))
    original = original_clusters[docs_here].values
    agree = (labels == original).sum()
    flipped = max(agree, len(labels) - agree)
    switched = [d for d, a, b in zip(docs_here, labels, original) if (a == b) != (agree >= len(labels) - agree)]
    print("   documents keeping their archetype:", flipped, "of", len(labels), " switched:", switched)

    print("GENERATIONS (paired countries present at this N):")
    for m in ["rho_OF_strict", "rho_OF_broad", "rho_FI_strict", "rho_FI_broad", "mediation_strict", "mediation_broad"]:
        changes = [sub.loc[b, m] - sub.loc[a, m] for _, a, b in PAIRS if a in sub.index and b in sub.index]
        down = sum(c < -1e-9 for c in changes)
        up = sum(c > 1e-9 for c in changes)
        print(f"   {m:17s} pairs={len(changes)} down={down} up={up} mean change={np.mean(changes):+.3f}")
