import os
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

codebook = yaml.safe_load(open("03_codebook.yaml"))
sections = ["objectives", "foresight", "instruments"]
codes = [(section, code) for section in sections for code in codebook[section]]

FACTS = [
    ("EU_2021", "instruments", "REGI", "above"),
    ("DEU_2018", "instruments", "FIN", "above"),
    ("FRA_2018", "objectives", "ETH", "above"),
    ("GBR_2025", "instruments", "INFRA", "above"),
    ("USA_2025", "instruments", "INFRA", "above"),
    ("USA_2025", "objectives", "ETH", "below"),
    ("CHN_2017", "objectives", "SEC", "above"),
]

coders = {}
for name, path in [("A", "data/04_coder_A.jsonl"), ("B", "data/05_coder_B.jsonl"), ("C", "data/06_coder_C.jsonl")]:
    if not os.path.exists(path):
        continue
    latest = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record["status"] == "ok":
                    latest[record["para_id"]] = record
    coders[name] = latest
    print(f"coder {name}: {len(latest)} paragraphs")

rows = []
for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
    if a not in coders or b not in coders:
        continue
    shared = sorted(set(coders[a]) & set(coders[b]))
    if not shared:
        continue
    for section, code in codes:
        x = np.array([code in coders[a][p][section] for p in shared], dtype=int)
        y = np.array([code in coders[b][p][section] for p in shared], dtype=int)
        agreement = float((x == y).mean())
        if x.sum() + y.sum() == 0 or len(set(x)) + len(set(y)) == 2 and set(x) == set(y):
            kappa = np.nan
        else:
            kappa = cohen_kappa_score(x, y)
        pi = (x.mean() + y.mean()) / 2
        expected = 2 * pi * (1 - pi)
        ac1 = (agreement - expected) / (1 - expected) if expected < 1 else np.nan
        rows.append({
            "pair": f"{a}-{b}",
            "section": section,
            "code": code,
            "n": len(shared),
            f"prev_first": round(float(x.mean()), 3),
            f"prev_second": round(float(y.mean()), 3),
            "agreement": round(agreement, 3),
            "kappa": round(float(kappa), 3) if not np.isnan(kappa) else np.nan,
            "ac1": round(float(ac1), 3) if not np.isnan(ac1) else np.nan,
        })

reliability = pd.DataFrame(rows)
reliability.to_csv("data/06_reliability.csv", index=False)

pd.set_option("display.width", 200)
print()
print("AGREEMENT PER CODE")
print(reliability.to_string(index=False))

print()
print("MEAN PER PAIR AND DIMENSION")
summary = reliability.groupby(["pair", "section"])[["agreement", "kappa", "ac1"]].mean().round(3)
print(summary.to_string())

fact_rows = []
for name, data in coders.items():
    records = list(data.values())
    for doc_id, section, code, direction in FACTS:
        overall = np.mean([code in r[section] for r in records]) if records else np.nan
        doc_records = [r for r in records if r["doc_id"] == doc_id]
        if not doc_records:
            fact_rows.append({"coder": name, "fact": f"{doc_id} {code} {direction}", "doc_share": np.nan, "overall_share": round(float(overall), 3), "result": "no data"})
            continue
        doc_share = np.mean([code in r[section] for r in doc_records])
        passed = doc_share > overall if direction == "above" else doc_share < overall
        fact_rows.append({"coder": name, "fact": f"{doc_id} {code} {direction}", "doc_share": round(float(doc_share), 3), "overall_share": round(float(overall), 3), "result": "PASS" if passed else "FAIL"})

facts = pd.DataFrame(fact_rows)
facts.to_csv("data/06_known_facts.csv", index=False)
print()
print("KNOWN FACTS CHECK")
print(facts.to_string(index=False))
