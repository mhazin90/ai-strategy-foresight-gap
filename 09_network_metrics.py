import sys
import numpy as np
import pandas as pd
import networkx as nx

THRESHOLD = int(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else 2
MIN_MENTIONS = int(sys.argv[sys.argv.index("--min-mentions") + 1]) if "--min-mentions" in sys.argv else 2
CODER = sys.argv[sys.argv.index("--coder") + 1] if "--coder" in sys.argv else "A"
suffix = f"_{CODER}_T{THRESHOLD}_M{MIN_MENTIONS}"

elements = pd.read_csv("data/07_elements.csv")
cells = pd.read_csv("data/07_cells.csv")
inventory = pd.read_csv("data/01_inventory.csv").set_index("id")
elements = elements[elements.coder == CODER]
cells = cells[(cells.coder == CODER) & (cells.max_strength >= THRESHOLD)]

doc_rows = []
node_rows = []
for doc_id, doc_elements in elements.groupby("doc_id"):
    present = doc_elements[doc_elements.n_paragraphs >= MIN_MENTIONS]
    node_type = dict(zip(present.code, present.section))
    objectives = [c for c, s in node_type.items() if s == "objectives"]
    foresight = [c for c, s in node_type.items() if s == "foresight"]
    instruments = [c for c, s in node_type.items() if s == "instruments"]

    graph = nx.Graph()
    graph.add_nodes_from(node_type)
    doc_cells = cells[cells.doc_id == doc_id]
    for source, target in zip(doc_cells.source, doc_cells.target):
        if source in node_type and target in node_type:
            graph.add_edge(source, target)

    betweenness = nx.betweenness_centrality(graph, normalized=True)
    degree = nx.degree_centrality(graph)

    of_pairs = {(s, t) for s, t in graph.edges() if node_type[s] == "objectives" and node_type[t] == "foresight"} | {(t, s) for s, t in graph.edges() if node_type[t] == "objectives" and node_type[s] == "foresight"}
    fi_pairs = {(s, t) for s, t in graph.edges() if node_type[s] == "foresight" and node_type[t] == "instruments"} | {(t, s) for s, t in graph.edges() if node_type[t] == "foresight" and node_type[s] == "instruments"}
    mediated = sum(1 for o in objectives for i in instruments if any((o, f) in of_pairs and (f, i) in fi_pairs for f in foresight))
    possible_pairs = len(objectives) * len(instruments)

    type_betweenness = {}
    for section in ["objectives", "foresight", "instruments"]:
        values = [betweenness[c] for c, s in node_type.items() if s == section]
        type_betweenness[section] = float(np.mean(values)) if values else np.nan

    doc_rows.append({
        "doc_id": doc_id,
        "generation": int(inventory.loc[doc_id, "generation"]),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "network_density": nx.density(graph) if graph.number_of_nodes() > 1 else np.nan,
        "clustering": nx.average_clustering(graph) if graph.number_of_nodes() > 0 else np.nan,
        "components": nx.number_connected_components(graph) if graph.number_of_nodes() > 0 else 0,
        "isolated_nodes": nx.number_of_isolates(graph),
        "betweenness_O": type_betweenness["objectives"],
        "betweenness_F": type_betweenness["foresight"],
        "betweenness_I": type_betweenness["instruments"],
        "mediation": mediated / possible_pairs if possible_pairs else np.nan,
    })

    for code, section in node_type.items():
        node_rows.append({"doc_id": doc_id, "code": code, "section": section, "degree": degree[code], "betweenness": betweenness[code]})

docs = pd.DataFrame(doc_rows)
nodes = pd.DataFrame(node_rows)
docs.to_csv(f"data/09_network_docs{suffix}.csv", index=False)
nodes.to_csv(f"data/09_network_nodes{suffix}.csv", index=False)

pd.set_option("display.width", 200)
print(f"CODER {CODER}  T={THRESHOLD}  M={MIN_MENTIONS}")
print(docs.round(3).to_string(index=False))
print()
print("MEANS")
print(docs[["network_density", "clustering", "betweenness_O", "betweenness_F", "betweenness_I", "mediation"]].mean().round(3).to_string())
print()
print("MOST CENTRAL NODES (mean betweenness across documents)")
print(nodes.groupby(["section", "code"])[["degree", "betweenness"]].mean().round(3).sort_values("betweenness", ascending=False).head(10).to_string())
