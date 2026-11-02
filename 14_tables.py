import json
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

inventory = pd.read_csv("data/01_inventory.csv").set_index("id")
documents = pd.read_csv("data/02_documents.csv").set_index("id")
measures = pd.read_csv("data/12_doc_measures.csv").set_index("doc_id")
reliability = pd.read_csv("data/06_reliability.csv")
summary = pd.read_csv("data/12_summary.csv")
facts = pd.read_csv("data/06_known_facts.csv")
rare = pd.read_csv("data/10b_rarefied.csv")

languages = {"BRA_2021": "Portuguese", "FRA_2021": "French", "JPN_2022": "English (machine translation)"}
table1 = pd.DataFrame({
    "country": inventory.loc[measures.index, "country"],
    "year": inventory.loc[measures.index, "year"],
    "generation": inventory.loc[measures.index, "generation"],
    "language": [languages.get(d, "English") for d in measures.index],
    "pages": inventory.loc[measures.index, "pages"],
    "paragraphs": measures["paragraphs"],
}).sort_values(["generation", "country"])
table1.to_csv("data/14_table1_sample.csv")
print("TABLE 1 SAMPLE")
print(table1.to_string())
print("totals: documents", len(table1), " pages", int(table1.pages.sum()), " paragraphs", int(table1.paragraphs.sum()))

detector = {}
for tag in ["A", "C"]:
    detector[tag] = {}
    with open(f"data/04c_foresight_{tag}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["status"] == "ok":
                detector[tag][r["para_id"]] = r["foresight"]
shared = sorted(set(detector["A"]) & set(detector["C"]))
a = np.array([int(detector["A"][p]) for p in shared])
c = np.array([int(detector["C"][p]) for p in shared])
agreement = float((a == c).mean())
pa, pc = a.mean(), c.mean()
expected = pa * pc + (1 - pa) * (1 - pc)
kappa_detector = (agreement - expected) / (1 - expected)

ab = reliability[reliability.pair == "A-B"].set_index("code")
ac = reliability[reliability.pair == "A-C"].set_index("code")
table2 = pd.DataFrame({
    "section": ac["section"],
    "prev_A_%": (100 * ac["prev_first"]).round(1),
    "prev_C_%": (100 * ac["prev_second"]).round(1),
    "kappa_A_B": ab["kappa"].round(2),
    "kappa_A_C": ac["kappa"].round(2),
    "agreement_A_C": ac["agreement"].round(3),
})
table2.to_csv("data/14_table2_reliability.csv")
print()
print("TABLE 2 RELIABILITY (A-C on", int(ac["n"].iloc[0]), "paragraphs)")
print(table2.to_string())
print("section means kappa A-C:", table2.groupby("section")["kappa_A_C"].mean().round(2).to_dict())
print("section means kappa A-B:", table2.groupby("section")["kappa_A_B"].mean().round(2).to_dict())
print(f"foresight detector A-C: n={len(shared)} prev_A={100 * pa:.1f}% prev_C={100 * pc:.1f}% agreement={agreement:.3f} kappa={kappa_detector:.2f}")

table3 = measures[["generation", "paragraphs", "rho_OI", "F_strict", "F_broad", "rho_OF_strict", "rho_OF_broad", "rho_FI_strict", "rho_FI_broad", "mediation_strict", "mediation_broad"]].copy()
table3 = table3.sort_values(["generation", "rho_OI"], ascending=[True, False]).round(2)
table3.to_csv("data/14_table3_results.csv")
print()
print("TABLE 3 MAIN RESULTS PER DOCUMENT")
print(table3.to_string())
print("means:", table3.drop(columns=["generation", "paragraphs"]).mean().round(3).to_dict())
print("documents without foresight  strict:", int((table3.F_strict == 0).sum()), " broad:", int((table3.F_broad == 0).sum()))

table4 = summary.copy()
table4.to_csv("data/14_table4_generations.csv", index=False)
print()
print("TABLE 4 GENERATION COMPARISON (8 paired countries)")
print(table4.to_string(index=False))

print()
print("TABLE S1 KNOWN FACTS")
print(facts.to_string(index=False))
print("pass rate per coder:", facts[facts.result != "no data"].groupby("coder")["result"].apply(lambda s: f"{(s == 'PASS').sum()}/{len(s)}").to_dict())

full = measures[["rho_OI", "rho_OF_strict", "rho_OF_broad", "rho_FI_strict", "rho_FI_broad", "mediation_strict", "mediation_broad"]].mean()
rows = [("Full length", len(measures), full)]
for n in sorted(rare.N.unique(), reverse=True):
    sub = rare[rare.N == n]
    rows.append((f"Rarefied N={n}", len(sub), sub[full.index].mean()))
tableS2 = pd.DataFrame([dict(condition=name, documents=count, **values.round(3).to_dict()) for name, count, values in rows])
tableS2.to_csv("data/14_tableS2_rarefaction.csv", index=False)
print()
print("TABLE S2 RAREFACTION")
print(tableS2.to_string(index=False))
