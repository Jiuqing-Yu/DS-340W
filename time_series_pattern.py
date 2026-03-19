
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from tslearn.metrics import cdist_dtw
import umap
from infomap import Infomap
import networkx as nx
from sklearn.metrics import silhouette_score

# -----------------------------
# 1) Load tidy dataset
# -----------------------------
df = pd.read_csv("Valve_Player_Data.csv")  # must have Game_Name, month, Avg_players
df = df.sort_values(["Game_Name", "Month_Year"])

games = df["Game_Name"].unique()
months = np.sort(df["Month_Year"].unique())

g2i = {g:i for i,g in enumerate(games)}
m2i = {m:i for i,m in enumerate(months)}

# Build matrix (#games × #months)
X = np.full((len(games), len(months)), np.nan)
for _, row in df.iterrows():
    X[g2i[row["Game_Name"]], m2i[row["Month_Year"]]] = row["Avg_players"]

# -----------------------------
# 2) Fill missing → z-score → smooth
# -----------------------------
def process_row(x):
    idx = np.arange(len(x))
    mask = ~np.isnan(x)
    if mask.sum() == 0:
        x = np.zeros_like(x)
    elif mask.sum() == 1:
        x[:] = x[mask][0]
    else:
        x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    x = zscore(x, nan_policy="omit")
    x = np.nan_to_num(x)
    return gaussian_filter1d(x, sigma=2)

X_proc = np.vstack([process_row(row) for row in X])

# -----------------------------
# 3) DTW distance matrix
# -----------------------------
DM = cdist_dtw(X_proc, X_proc)

# -----------------------------
# 4) UMAP → extract graph
# -----------------------------
um = umap.UMAP(
    n_neighbors=25,
    metric="precomputed",
    random_state=42
)
um.fit(DM)

# UMAP’s internal fuzzy graph
graph = um.graph_
print(graph)
# Convert UMAP fuzzy graph -> NetworkX 
G = nx.from_scipy_sparse_array(um.graph_, edge_attribute="weight")

# -----------------------------
# 5) Infomap clustering
# -----------------------------

im = Infomap(silent=True)
im.add_networkx_graph(G, weight='weight')
im.run()

# Cluster labels as a 1D numpy array aligned with `games`
modules = im.get_modules()  # node_id -> module_id
labels  = np.array([modules[i] for i in range(len(games))], dtype=int)

# 6) Evaluate
# -----------------------------
sil = silhouette_score(DM, labels, metric="precomputed")
print("Silhouette:", sil, "Clusters:", np.unique(labels))


# Save output
pd.DataFrame({"Game_Name": games, "cluster": labels}).to_csv(
    "game_clusters_simple.csv", index=False
)

# 7) PLOT CLUSTER SHAPES
# -----------------------------
import matplotlib.pyplot as plt

clusters = np.unique(labels)

# Normalize time axis between 0 and 1 for plotting
t = np.linspace(0, 1, X_proc.shape[1])

plt.figure(figsize=(10, 6))

for c in clusters:
    idx = np.where(labels == c)[0]
    curves = X_proc[idx]

    mean_curve = curves.mean(axis=0)
    std_curve  = curves.std(axis=0)

    plt.plot(t, mean_curve, label=f"Cluster {c} (n={len(idx)})")
    plt.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

plt.title("Cluster Mean Shapes (DTW + UMAP + Infomap)")
plt.xlabel("Normalized Time")
plt.ylabel("Smoothed Z-Score (Avg Players)")
plt.legend()
plt.tight_layout()
plt.show()