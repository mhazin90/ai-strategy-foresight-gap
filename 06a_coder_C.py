import os
import sys
import json
import time
import yaml
import pandas as pd
import ollama

MODEL = "llama3.1:8b"
TOTAL_PARAGRAPHS = len(pd.read_csv("data/02_paragraphs.csv"))

codebook = yaml.safe_load(open("03_codebook.yaml"))
paragraphs = pd.read_csv("data/02_paragraphs.csv")

test_n = int(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else 0
out_path = "data/06_test_C.jsonl" if test_n else "data/06_coder_C.jsonl"
if test_n:
    paragraphs = paragraphs.sample(n=test_n, random_state=42)
else:
    paragraphs = paragraphs.sample(n=500, random_state=11)

lines = ["You code paragraphs of national AI strategy documents using a fixed codebook.", "", "GENERAL RULES"]
lines += [f"- {rule}" for rule in codebook["general_rules"]]
lines += ["", "STRENGTH SCALE FOR LINKS"]
lines += [f"{level}: {text}" for level, text in codebook["strength_scale"].items()]
for section in ["objectives", "foresight", "instruments"]:
    lines += ["", section.upper()]
    for code, item in codebook[section].items():
        lines.append(f"{code} ({item['name']}): {item['definition']}")
lines += ["", "LINK TYPES"]
lines += [f"{link_type}: {text}" for link_type, text in codebook["link_types"].items()]
lines += [
    "",
    "Return only JSON with exactly this structure:",
    '{"objectives": [codes], "foresight": [codes], "instruments": [codes], "links": [{"type": "OF or FI or OI", "source": "code", "target": "code", "strength": 1, "evidence": "shortest exact quote, at most 20 words"}]}',
    "OF links go from an objective code to a foresight code. FI links go from a foresight code to an instrument code. OI links go from an objective code to an instrument code.",
    "If nothing applies, return empty lists.",
    "The paragraph may be in English, French, or Portuguese. Always answer with the codes above and quote evidence in the original language.",
]
system_prompt = "\n".join(lines)

valid = {section: set(codebook[section]) for section in ["objectives", "foresight", "instruments"]}
link_dims = {"OF": ("objectives", "foresight"), "FI": ("foresight", "instruments"), "OI": ("objectives", "instruments")}

done = set()
if os.path.exists(out_path):
    with open(out_path) as f:
        done = {json.loads(line)["para_id"] for line in f if line.strip() and json.loads(line)["status"] == "ok"}
todo = paragraphs[~paragraphs["para_id"].isin(done)]
print(f"to code: {len(todo)}  already done: {len(done)}", flush=True)

with open(out_path, "a") as out:
    for k, (_, row) in enumerate(todo.iterrows(), start=1):
        t0 = time.time()
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "PARAGRAPH:\n" + row["text"]},
                ],
                format="json",
                options={"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 350},
            )
            raw = response["message"]["content"]
            parsed = json.loads(raw)
            status = "ok"
        except Exception as e:
            raw, parsed, status = "", {}, "error_" + type(e).__name__

        result = {}
        for section in valid:
            items = parsed.get(section, []) if isinstance(parsed.get(section, []), list) else []
            result[section] = {c for c in items if isinstance(c, str) and c in valid[section]}

        best_links = {}
        raw_links = parsed.get("links", []) if isinstance(parsed.get("links", []), list) else []
        paragraph_norm = " ".join(row["text"].lower().split())
        for link in raw_links:
            if not isinstance(link, dict) or link.get("type") not in link_dims:
                continue
            source_dim, target_dim = link_dims[link["type"]]
            source, target = link.get("source"), link.get("target")
            if source not in valid[source_dim] or target not in valid[target_dim]:
                continue
            try:
                strength = int(link.get("strength"))
            except Exception:
                continue
            if strength not in (1, 2, 3):
                continue
            result[source_dim].add(source)
            result[target_dim].add(target)
            evidence = str(link.get("evidence", ""))
            evidence_norm = " ".join(evidence.lower().split())
            key = (link["type"], source, target)
            if key not in best_links or strength > best_links[key]["strength"]:
                best_links[key] = {
                    "type": link["type"],
                    "source": source,
                    "target": target,
                    "strength": strength,
                    "evidence": evidence,
                    "evidence_found": len(evidence_norm) > 0 and evidence_norm[:40] in paragraph_norm,
                }

        record = {
            "para_id": row["para_id"],
            "doc_id": row["doc_id"],
            "status": status,
            "objectives": sorted(result["objectives"]),
            "foresight": sorted(result["foresight"]),
            "instruments": sorted(result["instruments"]),
            "links": list(best_links.values()),
            "seconds": round(time.time() - t0, 1),
            "raw": raw,
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{k}/{len(todo)} {row['para_id']} {record['seconds']}s O={record['objectives']} F={record['foresight']} I={record['instruments']} links={len(record['links'])}", flush=True)

records = [json.loads(line) for line in open(out_path) if line.strip()]
ok = [r for r in records if r["status"] == "ok"]
all_links = [l for r in ok for l in r["links"]]
mean_seconds = sum(r["seconds"] for r in ok) / len(ok) if ok else 0

print()
print("SUMMARY")
print("coded:", len(records), " ok:", len(ok), " errors:", len(records) - len(ok))
print("paragraphs with any code:", sum(1 for r in ok if r["objectives"] or r["foresight"] or r["instruments"]))
print("links:", len(all_links), " evidence found in text:", sum(l["evidence_found"] for l in all_links))
print("mean seconds per paragraph:", round(mean_seconds, 1))
print("estimated hours for all paragraphs:", round(mean_seconds * TOTAL_PARAGRAPHS / 3600, 1))
