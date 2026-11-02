import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

B = 500
K_RANGE = range(2, 7)

docs = pd.read_csv("data/12_doc_measures.csv").set_index("doc_id")
docs["binding_ratio"] = docs["binding_ratio"].fillna(docs["binding_ratio"].mean())
for name in ["F_rate", "rho_OF", "rho_FI", "mediation"]:
    docs[f"{name}_mean"] = (docs[f"{name}_strict"] + docs[f"{name}_broad"]) / 2

feature_sets = {
    "mean": ["rho_OI", "F_rate_mean", "rho_OF_mean", "rho_FI_mean", "mediation_mean", "share_ETH", "binding_ratio"],
    "strict": ["rho_OI", "F_rate_strict", "rho_OF_strict", "rho_FI_strict", "mediation_strict", "share_ETH", "binding_ratio"],
    "broad": ["rho_OI", "F_rate_broad", "rho_OF_broad", "rho_FI_broad", "mediation_broad", "share_ETH", "binding_ratio"],
}

labels_by_set = {}
best_k = {}
for set_name, features in feature_sets.items():
    X = docs[features].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    scores = {}
    for k in K_RANGE:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k[set_name] = max(scores, key=scores.get)
    labels_by_set[set_name] = AgglomerativeClustering(n_clusters=best_k[set_name], linkage="ward").fit_predict(X)
    print(f"SILHOUETTE ({set_name}):", {k: round(v, 3) for k, v in scores.items()}, " best k:", best_k[set_name])

features = feature_sets["mean"]
X = docs[features].values
X = (X - X.mean(axis=0)) / X.std(axis=0)
k = best_k["mean"]
labels = labels_by_set["mean"]
n = len(X)

rng = np.random.default_rng(42)
jaccards = {c: [] for c in range(k)}
for b in range(B):
    idx = rng.choice(n, n, replace=True)
    unique_docs = np.unique(idx)
    if len(unique_docs) <= k:
        continue
    boot_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X[idx])
    label_of = {}
    for pos, d in enumerate(idx):
        label_of.setdefault(d, boot_labels[pos])
    boot_clusters = {}
    for d, lab in label_of.items():
        boot_clusters.setdefault(lab, set()).add(d)
    for c in range(k):
        original = set(np.where(labels == c)[0]) & set(unique_docs)
        if not original:
            continue
        jaccards[c].append(max(len(original & bc) / len(original | bc) for bc in boot_clusters.values()))

docs["cluster"] = labels
docs.to_csv("data/11_clusters.csv")

pd.set_option("display.width", 250)
print()
print(f"CHOSEN k = {k} (mean features)")
for c in range(k):
    members = list(docs.index[docs.cluster == c])
    stability = np.mean(jaccards[c]) if jaccards[c] else np.nan
    verdict = "stable" if stability >= 0.75 else ("weak pattern" if stability >= 0.6 else "not trustworthy")
    print(f"cluster {c}: n={len(members)}  bootstrap Jaccard={stability:.2f} ({verdict})")
    print("   members:", ", ".join(members))
print()
print("CLUSTER PROFILES (raw means)")
print(docs.groupby("cluster")[features].mean().round(3).to_string())
print()
print("ROBUSTNESS (adjusted Rand index, 1 = identical)")
print("mean vs strict:", round(adjusted_rand_score(labels_by_set["mean"], labels_by_set["strict"]), 3))
print("mean vs broad:", round(adjusted_rand_score(labels_by_set["mean"], labels_by_set["broad"]), 3))
print("strict vs broad:", round(adjusted_rand_score(labels_by_set["strict"], labels_by_set["broad"]), 3))
