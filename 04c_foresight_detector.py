import os
import re
import sys
import json
import time
import yaml
import pandas as pd
import ollama

MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "qwen2.5:7b"
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "A"
SAMPLE = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 0
CANDIDATES_ONLY = "--candidates-only" in sys.argv

codebook = yaml.safe_load(open("03_codebook.yaml"))
paragraphs = pd.read_csv("data/02_paragraphs.csv")

KEYWORDS = {
    "SCEN": [r"scenarios?", r"sc[ée]narios?", r"cen[áa]rios?", r"futures studies", r"prospective", r"prospec[çc][ãa]o", r"foresight"],
    "EXP": [r"expert (group|panel|committee|council|commission)s?", r"advisory (board|council|committee|body|group)s?", r"high[- ]level (expert )?group", r"delphi", r"comit[ée] d'experts", r"groupe d'experts", r"conseil consultatif", r"comit[êe] de especialistas", r"grupo de (especialistas|trabalho)", r"conselho consultivo"],
    "ROAD": [r"road ?maps?", r"feuille de route", r"roteiros?", r"technology roadmap"],
    "TREND": [r"horizon scanning", r"forecast(s|ing)?", r"observator(y|ies)", r"observatoire", r"observat[óo]rio", r"trend analysis", r"technology watch", r"veille", r"monitoring of (developments|trends|technolog)", r"indicators?", r"indicadores", r"indicateurs", r"anticipat(e|es|ed|ing|ion|ory)", r"anticip(er|ation)", r"antecipa(r|[çc][ãa]o)"],
    "RISK": [r"risk assessments?", r"impact assessments?", r"risk analys[ie]s", r"[ée]valuation des risques", r"[ée]tude d'impact", r"avalia[çc][ãa]o de (risco|impacto)", r"an[áa]lise de risco", r"(research|study|studies|assessing|assessment|review|reviewing) (into|of|on) the (impact|impacts|effects|risks)"],
    "STAKE": [r"public consultations?", r"consultations? with", r"stakeholder (consultation|dialogue|engagement)s?", r"citizens'? (panel|assembl(y|ies)|dialogue|jur(y|ies))", r"public dialogue", r"concertation", r"consultation publique", r"d[ée]bat public", r"consultas? p[úu]blicas?", r"di[áa]logo com", r"consult(s|ed|ing|ation|ations)?", r"consulter", r"consultar"],
}

matched = []
for text in paragraphs["text"].str.lower():
    hits = []
    for code, patterns in KEYWORDS.items():
        for pat in patterns:
            if re.search(r"\b" + pat + r"\b", text):
                hits.append(code)
                break
    matched.append(",".join(hits))
paragraphs["keyword_types"] = matched
candidates = paragraphs[paragraphs["keyword_types"] != ""].copy()
candidates[["para_id", "doc_id", "keyword_types"]].to_csv("data/04c_candidates.csv", index=False)

print("STAGE 1 KEYWORDS")
print("candidate paragraphs:", len(candidates), "of", len(paragraphs), f"({len(candidates) / len(paragraphs):.1%})")
type_counts = {code: int(candidates["keyword_types"].str.contains(code).sum()) for code in KEYWORDS}
print("candidates per type:", type_counts)

if os.path.exists("data/04_coder_A.jsonl") and os.path.exists("data/06_coder_C.jsonl"):
    coded = {}
    for name, path in [("A", "data/04_coder_A.jsonl"), ("C", "data/06_coder_C.jsonl")]:
        coded[name] = {}
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if record["status"] == "ok":
                    coded[name][record["para_id"]] = record
    agreed = [p for p in set(coded["A"]) & set(coded["C"]) if coded["A"][p]["foresight"] and coded["C"][p]["foresight"]]
    covered = [p for p in agreed if p in set(candidates["para_id"])]
    print("RECALL CHECK: paragraphs where A and C both coded foresight:", len(agreed), " of these with a keyword:", len(covered))

if CANDIDATES_ONLY:
    sys.exit(0)

if SAMPLE:
    candidates = candidates.sample(n=min(SAMPLE, len(candidates)), random_state=11)

objective_list = ", ".join(f"{c} ({i['name']})" for c, i in codebook["objectives"].items())
instrument_list = ", ".join(f"{c} ({i['name']})" for c, i in codebook["instruments"].items())
foresight_lines = [f"{c} ({i['name']}): {i['definition']}" for c, i in codebook["foresight"].items()]

system_prompt = "\n".join([
    "You decide whether a paragraph from a national AI strategy describes a concrete foresight activity that the authoring government or organisation itself carries out, has carried out to prepare this strategy, plans, or commissions.",
    "",
    "FORESIGHT ACTIVITY TYPES",
    *foresight_lines,
    "",
    "DO NOT COUNT",
    "- General statements about the future, about AI, or about trends.",
    "- Descriptions of technologies, markets, or statistics.",
    "- Citations of reports, studies, or scenarios produced by others, for example the OECD, the UN, consultancies, or other countries.",
    "- Activities of other countries or organisations, unless this government adopts them as its own.",
    "- Goals or visions that are not produced by an anticipatory process.",
    "- Monitoring of this strategy's own budget, progress, or implementation. That is evaluation, not foresight.",
    "",
    "For each real activity, also give the objectives it addresses and the instruments whose design it informs, using these codes.",
    "OBJECTIVES: " + objective_list,
    "INSTRUMENTS: " + instrument_list,
    "STRENGTH: 1 = only mentioned together, 2 = the text states the relation, 3 = stated relation plus a budget, named programme, deadline, or named responsible body.",
    "",
    "Return only JSON with exactly this structure:",
    '{"foresight": true or false, "activities": [{"type": "code", "evidence": "shortest exact quote, at most 20 words", "objectives": [{"code": "code", "strength": 1}], "instruments": [{"code": "code", "strength": 1}]}]}',
    "If there is no real foresight activity, return {\"foresight\": false, \"activities\": []}.",
    "The paragraph may be in English, French, or Portuguese. Quote evidence in the original language.",
])

valid_f = set(codebook["foresight"])
valid_o = set(codebook["objectives"])
valid_i = set(codebook["instruments"])
out_path = f"data/04c_foresight_{TAG}.jsonl"

done = set()
if os.path.exists(out_path):
    with open(out_path) as f:
        done = {json.loads(line)["para_id"] for line in f if line.strip() and json.loads(line)["status"] == "ok"}
todo = candidates[~candidates["para_id"].isin(done)]
print(f"STAGE 2 MODEL {MODEL}  to code: {len(todo)}  already done: {len(done)}", flush=True)

with open(out_path, "a") as out:
    for k, (_, row) in enumerate(todo.iterrows(), start=1):
        t0 = time.time()
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "PARAGRAPH:\n" + row["text"]}],
                format="json",
                options={"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 600},
            )
            raw = response["message"]["content"]
            parsed = json.loads(raw)
            status = "ok"
        except Exception as e:
            raw, parsed, status = "", {}, "error_" + type(e).__name__

        paragraph_norm = " ".join(row["text"].lower().split())
        activities = []
        raw_activities = parsed.get("activities", []) if isinstance(parsed.get("activities", []), list) else []
        for activity in raw_activities:
            if not isinstance(activity, dict) or activity.get("type") not in valid_f:
                continue
            evidence = str(activity.get("evidence", ""))
            evidence_norm = " ".join(evidence.lower().split())
            linked = {}
            for field, valid in [("objectives", valid_o), ("instruments", valid_i)]:
                items = activity.get(field, []) if isinstance(activity.get(field, []), list) else []
                linked[field] = []
                for item in items:
                    if not isinstance(item, dict) or item.get("code") not in valid:
                        continue
                    try:
                        strength = int(item.get("strength"))
                    except Exception:
                        continue
                    if strength in (1, 2, 3):
                        linked[field].append({"code": item["code"], "strength": strength})
            activities.append({
                "type": activity["type"],
                "evidence": evidence,
                "evidence_found": len(evidence_norm) > 0 and evidence_norm[:40] in paragraph_norm,
                "objectives": linked["objectives"],
                "instruments": linked["instruments"],
            })

        record = {
            "para_id": row["para_id"],
            "doc_id": row["doc_id"],
            "status": status,
            "foresight": bool(activities),
            "activities": activities,
            "seconds": round(time.time() - t0, 1),
            "raw": raw,
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{k}/{len(todo)} {row['para_id']} {record['seconds']}s foresight={record['foresight']} types={[a['type'] for a in activities]}", flush=True)

records = {}
with open(out_path) as f:
    for line in f:
        record = json.loads(line)
        records[record["para_id"]] = record
ok = [r for r in records.values() if r["status"] == "ok"]
acts = [a for r in ok for a in r["activities"]]
print()
print("SUMMARY")
print("candidates coded:", len(records), " ok:", len(ok), " errors:", len(records) - len(ok))
print("with real foresight:", sum(r["foresight"] for r in ok), f"({sum(r['foresight'] for r in ok) / max(len(ok), 1):.1%} of candidates)")
print("activities per type:", {t: sum(a["type"] == t for a in acts) for t in sorted(valid_f)})
print("evidence found in text:", sum(a["evidence_found"] for a in acts), "of", len(acts))
