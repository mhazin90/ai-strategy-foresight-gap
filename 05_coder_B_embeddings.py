import re
import json
import yaml
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
Z_HIGH = 1.5
Z_LOW = 0.0

codebook = yaml.safe_load(open("03_codebook.yaml"))
paragraphs = pd.read_csv("data/02_paragraphs.csv")
sections = ["objectives", "foresight", "instruments"]
link_dims = {"OF": ("objectives", "foresight"), "FI": ("foresight", "instruments"), "OI": ("objectives", "instruments")}

COMMITMENT = re.compile(r"\b(will|shall|plan|plans|planned|aim|aims|propose|proposes|launch|launches|establish|establishes|set up|allocate|allocates|invest|invests|fund|funds|create|creates|introduce|introduces|expand|expands|implement|va|vont|sera|seront|lancer|créer|financer|mettre en place|será|serão|irá|irão|criar|implementar|investir|lançar)\b")
RELATION = re.compile(r"\b(to support|in order to|aimed at|aims to|to promote|to ensure|to strengthen|to achieve|to enable|by means of|based on|informed by|afin de|visant à|en vue de|a fim de|com o objetivo de|por meio de|visando)\b")
SPECIFIC = re.compile(r"(\d[\d.,]*\s*(million|billion|bn|mrd|milliards?|milhões|bilhões|crore|lakh)\b|[€$£¥]\s?\d|\b20[2-4]\d\b|\b(ministry|agency|office|institute|council|commission|department|ministère|agence|ministério|agência)\b)")

codes = []
section_of = {}
descriptions = []
keyword_patterns = {}
for section in sections:
    for code, item in codebook[section].items():
        codes.append(code)
        section_of[code] = section
        descriptions.append(f"{item['name']}. {item['definition']}")
        keyword_patterns[code] = re.compile(r"\b(" + "|".join(re.escape(k.lower()) for k in item["keywords"]) + r")\b")

model = SentenceTransformer(MODEL_NAME)
code_embeddings = model.encode(descriptions, normalize_embeddings=True)
paragraph_embeddings = model.encode(paragraphs["text"].tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=True)
similarity = paragraph_embeddings @ code_embeddings.T
z_scores = (similarity - similarity.mean(axis=0)) / similarity.std(axis=0)

similarity_df = pd.DataFrame(similarity, columns=codes)
similarity_df.insert(0, "para_id", paragraphs["para_id"].values)
similarity_df.to_csv("data/05_similarity.csv", index=False)

records = []
for i, row in paragraphs.iterrows():
    text = row["text"].lower()
    has_commitment = bool(COMMITMENT.search(text))
    has_relation = bool(RELATION.search(text))
    has_specific = bool(SPECIFIC.search(text))

    present = {section: [] for section in sections}
    for j, code in enumerate(codes):
        score = z_scores[i, j]
        keyword_hit = bool(keyword_patterns[code].search(text))
        is_present = score >= Z_HIGH or (keyword_hit and score >= Z_LOW)
        if section_of[code] == "instruments" and not has_commitment:
            is_present = False
        if is_present:
            present[section_of[code]].append(code)

    if has_relation and has_specific:
        strength = 3
    elif has_relation:
        strength = 2
    else:
        strength = 1

    links = []
    for link_type, (source_dim, target_dim) in link_dims.items():
        for source in present[source_dim]:
            for target in present[target_dim]:
                links.append({"type": link_type, "source": source, "target": target, "strength": strength})

    records.append({
        "para_id": row["para_id"],
        "doc_id": row["doc_id"],
        "status": "ok",
        "objectives": sorted(present["objectives"]),
        "foresight": sorted(present["foresight"]),
        "instruments": sorted(present["instruments"]),
        "links": links,
        "commitment": has_commitment,
        "relation": has_relation,
        "specific": has_specific,
    })

with open("data/05_coder_B.jsonl", "w") as out:
    for record in records:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")

print()
print("SIMILARITY PERCENTILES (all codes)")
print("p50:", round(float(np.percentile(similarity, 50)), 3), " p90:", round(float(np.percentile(similarity, 90)), 3), " p99:", round(float(np.percentile(similarity, 99)), 3))

print()
print("SHARE OF PARAGRAPHS PER CODE")
counts = Counter(code for r in records for section in sections for code in r[section])
for code in codes:
    print(f"{code:6s} {section_of[code]:12s} {counts[code] / len(records):.2f}")

print()
print("paragraphs:", len(records))
print("with any code:", sum(1 for r in records if r["objectives"] or r["foresight"] or r["instruments"]))
print("links:", sum(len(r["links"]) for r in records))

print()
print("SAME 20 TEST PARAGRAPHS AS CODER A")
test_ids = set(paragraphs.sample(n=20, random_state=42)["para_id"])
for r in records:
    if r["para_id"] in test_ids:
        print(r["para_id"].ljust(15), r["objectives"], r["foresight"], r["instruments"])
