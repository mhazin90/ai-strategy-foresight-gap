import sys
import yaml
import numpy as np
import pandas as pd

THRESHOLD = int(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else 2
MIN_MENTIONS = int(sys.argv[sys.argv.index("--min-mentions") + 1]) if "--min-mentions" in sys.argv else 2
suffix = f"_T{THRESHOLD}_M{MIN_MENTIONS}"

codebook = yaml.safe_load(open("03_codebook.yaml"))
objective_codes = list(codebook["objectives"])
elements = pd.read_csv("data/07_elements.csv")
cells = pd.read_csv("data/07_cells.csv")
inventory = pd.read_csv("data/01_inventory.csv").set_index("id")

rows = []
closure_rows = []
for (coder, doc_id), doc_elements in elements.groupby(["coder", "doc_id"]):
    present = doc_elements[doc_elements.n_paragraphs >= MIN_MENTIONS]
    objectives = present[present.section == "objectives"].code.tolist()
    foresight = present[present.section == "foresight"].code.tolist()
    instruments = present[present.section == "instruments"].code.tolist()

    doc_cells = cells[(cells.coder == coder) & (cells.doc_id == doc_id) & (cells.max_strength >= THRESHOLD)]
    of_links = {(s, t) for s, t in zip(doc_cells[doc_cells.type == "OF"].source, doc_cells[doc_cells.type == "OF"].target) if s in objectives and t in foresight}
    fi_links = {(s, t) for s, t in zip(doc_cells[doc_cells.type == "FI"].source, doc_cells[doc_cells.type == "FI"].target) if s in foresight and t in instruments}
    oi_links = {(s, t) for s, t in zip(doc_cells[doc_cells.type == "OI"].source, doc_cells[doc_cells.type == "OI"].target) if s in objectives and t in instruments}

    closures = [(o, f, i) for o in objectives for f in foresight for i in instruments if (o, f) in of_links and (f, i) in fi_links and (o, i) in oi_links]
    n_o, n_f, n_i = len(objectives), len(foresight), len(instruments)
    possible = n_o * n_f * n_i

    row = {
        "coder": coder,
        "doc_id": doc_id,
        "generation": int(inventory.loc[doc_id, "generation"]),
        "n_O": n_o,
        "n_F": n_f,
        "n_I": n_i,
        "closures": len(closures),
        "possible": possible,
        "density": len(closures) / possible if possible else np.nan,
        "rho_OF": len(of_links) / (n_o * n_f) if n_o * n_f else np.nan,
        "rho_FI": len(fi_links) / (n_f * n_i) if n_f * n_i else np.nan,
        "rho_OI": len(oi_links) / (n_o * n_i) if n_o * n_i else np.nan,
    }
    for code in objective_codes:
        row[f"closures_{code}"] = sum(1 for o, _, _ in closures if o == code)
    rows.append(row)

    for o, f, i in closures:
        closure_rows.append({"coder": coder, "doc_id": doc_id, "objective": o, "foresight": f, "instrument": i})

results = pd.DataFrame(rows)
results.to_csv(f"data/08_closures{suffix}.csv", index=False)
pd.DataFrame(closure_rows).to_csv(f"data/08_closure_list{suffix}.csv", index=False)

pd.set_option("display.width", 200)
print(f"THRESHOLD T={THRESHOLD}  MIN_MENTIONS M={MIN_MENTIONS}")
for coder in results.coder.unique():
    print()
    print("CODER", coder)
    table = results[results.coder == coder][["doc_id", "generation", "n_O", "n_F", "n_I", "closures", "possible", "density", "rho_OF", "rho_FI", "rho_OI"]]
    print(table.round(3).to_string(index=False))
    print("means:", table[["density", "rho_OF", "rho_FI", "rho_OI"]].mean().round(3).to_dict())
