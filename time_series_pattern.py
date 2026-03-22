
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from tslearn.metrics import cdist_dtw
import umap
from infomap import Infomap
import networkx as nx
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# -----------------------------
# 1) Load tidy dataset
# -----------------------------
df = pd.read_csv("Valve_Player_Data_with_genres.csv")  # must have Game_Name, month, Avg_players
df = df.sort_values(["Game_Name", "Month_Year"])
print([col for col in df.columns if "genre" in col])
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
    n_neighbors=15,
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
embedding = um.embedding_
sil = silhouette_score(embedding, labels)
print("Silhouette:", sil, "Clusters:", np.unique(labels))


# Save output
pd.DataFrame({"Game_Name": games, "cluster": labels}).to_csv(
    "game_clusters_simple.csv", index=False
)

# -----------------------------
# 7) Plot cluster shapes
# -----------------------------
clusters = np.unique(labels)
t = np.linspace(0, 1, X_proc.shape[1])

for c in clusters:
    plt.figure(figsize=(8, 5))
    idx = np.where(labels == c)[0]
    curves = X_proc[idx]

    mean_curve = curves.mean(axis=0)
    std_curve  = curves.std(axis=0)

    # plot individual curves
    for curve in curves:
        plt.plot(t, curve, alpha=0.2)

    # plot mean + std
    plt.plot(t, mean_curve, linewidth=2, label="Mean")
    plt.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)

    plt.title(f"Cluster {c} (n={len(idx)})")
    plt.xlabel("Normalized Time")
    plt.ylabel("Smoothed Z-Score (Avg Players)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# 8) Plot genre distribution per cluster
# -----------------------------
# Identify genre columns
genre_cols = [col for col in df.columns if col.startswith("genre_")]

# One row per game
game_genres = df.drop_duplicates("Game_Name")[["Game_Name"] + genre_cols]

# Merge with cluster labels
cluster_df = pd.DataFrame({"Game_Name": games, "cluster": labels})
merged = cluster_df.merge(game_genres, on="Game_Name", how="left")

clusters = sorted(merged["cluster"].unique())

# Compute total number of games per genre (denominator for all clusters)
total_genre_counts = merged[genre_cols].sum()

# Loop over clusters → one plot per cluster
for c in clusters:
    subset = merged[merged["cluster"] == c]
    
    if subset.shape[0] == 0:
        print(f"Cluster {c} has no games, skipping.")
        continue
    
    # Count how many games in this cluster have each genre
    cluster_genre_counts = subset[genre_cols].sum()
    
    # Compute percentage relative to total games with that genre
    genre_percent = cluster_genre_counts / total_genre_counts * 100
    genre_percent = genre_percent.fillna(0)
    
    # Skip cluster if all zeros
    if genre_percent.sum() == 0:
        print(f"Cluster {c} has no genre flags, skipping.")
        continue
    
    # Plot
    plt.figure(figsize=(10,5))
    genre_percent.plot(kind="bar", color="skyblue")
    plt.title(f"Cluster {c} - % of each genre in this cluster (relative to all games with genre)")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Genre")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()