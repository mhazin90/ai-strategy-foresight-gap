import os
import json
import yaml
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_COVERAGE = 0.95

codebook = yaml.safe_load(open("03_codebook.yaml"))
sections = ["objectives", "foresight", "instruments"]
codes = [(section, code) for section in sections for code in codebook[section]]
paragraphs = pd.read_csv("data/02_paragraphs.csv")
doc_sizes = paragraphs.groupby("doc_id").size()

raw = {}
for name, path in [("A", "data/04_coder_A.jsonl"), ("B", "data/05_coder_B.jsonl")]:
    if not os.path.exists(path):
        continue
    latest = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record["status"] == "ok":
                    latest[record["para_id"]] = record
    raw[name] = latest

coders = dict(raw)
if "A" in raw and "B" in raw:
    consensus = {}
    for para_id in set(raw["A"]) & set(raw["B"]):
        record_a = raw["A"][para_id]
        record_b = raw["B"][para_id]
        merged = {"para_id": para_id, "doc_id": record_a["doc_id"]}
        for section in sections:
            merged[section] = sorted(set(record_a[section]) & set(record_b[section]))
        b_strength = {(l["type"], l["source"], l["target"]): l["strength"] for l in record_b["links"]}
        merged_links = []
        for link in record_a["links"]:
            key = (link["type"], link["source"], link["target"])
            if key in b_strength:
                merged_links.append({"type": link["type"], "source": link["source"], "target": link["target"], "strength": min(link["strength"], b_strength[key])})
        merged["links"] = merged_links
        consensus[para_id] = merged
    coders["AB"] = consensus

element_rows = []
cell_rows = []
complete = {}
for name, data in coders.items():
    by_doc = {}
    for record in data.values():
        by_doc.setdefault(record["doc_id"], []).append(record)
    complete[name] = [d for d in doc_sizes.index if len(by_doc.get(d, [])) / doc_sizes[d] >= MIN_COVERAGE]
    print(f"coder {name}: {len(complete[name])} of {len(doc_sizes)} documents complete")

    for doc_id in complete[name]:
        records = by_doc[doc_id]
        n_doc = doc_sizes[doc_id]
        for section, code in codes:
            count = sum(code in r[section] for r in records)
            element_rows.append({"coder": name, "doc_id": doc_id, "section": section, "code": code, "n_paragraphs": count, "share": count / n_doc})

        cells = {}
        for record in records:
            for link in record["links"]:
                key = (link["type"], link["source"], link["target"])
                if key not in cells:
                    cells[key] = {"max_strength": 0, "n_links": 0, "n_strong": 0}
                cells[key]["max_strength"] = max(cells[key]["max_strength"], link["strength"])
                cells[key]["n_links"] += 1
                cells[key]["n_strong"] += int(link["strength"] >= 2)
        for (link_type, source, target), values in cells.items():
            cell_rows.append({"coder": name, "doc_id": doc_id, "type": link_type, "source": source, "target": target, **values})

elements = pd.DataFrame(element_rows)
cells = pd.DataFrame(cell_rows)
elements.to_csv("data/07_elements.csv", index=False)
cells.to_csv("data/07_cells.csv", index=False)
print()
print("element rows:", len(elements), " cell rows:", len(cells))

if "A" in complete and "B" in complete:
    shared_docs = sorted(set(complete["A"]) & set(complete["B"]))
    print()
    print("DOCUMENT-LEVEL AGREEMENT A vs B on", len(shared_docs), "documents")
    if len(shared_docs) >= 5:
        agreement_rows = []
        for section, code in codes:
            share_a = elements[(elements.coder == "A") & (elements.code == code) & (elements.doc_id.isin(shared_docs))].set_index("doc_id")["share"].reindex(shared_docs)
            share_b = elements[(elements.coder == "B") & (elements.code == code) & (elements.doc_id.isin(shared_docs))].set_index("doc_id")["share"].reindex(shared_docs)
            rho = spearmanr(share_a, share_b).statistic if share_a.std() > 0 and share_b.std() > 0 else np.nan
            agreement_rows.append({"section": section, "code": code, "spearman": round(float(rho), 3) if not np.isnan(rho) else np.nan})
        doc_agreement = pd.DataFrame(agreement_rows)
        doc_agreement.to_csv("data/07_doc_agreement.csv", index=False)
        print(doc_agreement.to_string(index=False))
        print()
        print(doc_agreement.groupby("section")["spearman"].mean().round(3).to_string())

        jaccards = []
        for doc_id in shared_docs:
            strong_a = set(map(tuple, cells[(cells.coder == "A") & (cells.doc_id == doc_id) & (cells.max_strength >= 2)][["type", "source", "target"]].values))
            strong_b = set(map(tuple, cells[(cells.coder == "B") & (cells.doc_id == doc_id) & (cells.max_strength >= 2)][["type", "source", "target"]].values))
            union = strong_a | strong_b
            jaccards.append(len(strong_a & strong_b) / len(union) if union else np.nan)
        print()
        print("mean Jaccard of strong cells (strength >= 2) per document:", round(float(np.nanmean(jaccards)), 3))
    else:
        print("not enough complete documents yet")
