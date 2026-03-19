import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
# Load data
df = pd.read_excel(
    "RESULTS - Real Skills in a Virtual World - The Role of Video Game Genre Preferences in Young Adults' Social and Emotional Competence.xlsx",
    sheet_name="CLEAN",
    header=1,
    engine="openpyxl"
)

# One-hot encode gender
df = pd.concat([
    df,
    pd.get_dummies(df["Please specify your gender. - Selected Choice"])
], axis=1)
#
df = pd.concat([
    df,
    pd.get_dummies(df["How often do you turn to video games as a form of escape from your daily responsibilities?"])
], axis=1)
print(df["How often do you turn to video games as a form of escape from your daily responsibilities?"].unique())
# ONE-HOT ENCODE AGE GROUPS
df["Age"] = pd.to_numeric(df["Please indicate your age (enter a two-digit number, such as 20)."], errors="coerce")
bins = [18, 25, 33, 100]
labels = ["18-24", "25-32", "33+"]

df["Age_group"] = pd.cut(
    df["Age"],
    bins=bins,
    labels=labels,
    right=False
)

age_dummies = pd.get_dummies(df["Age_group"])
df = pd.concat([df, age_dummies], axis=1)

# LEFT SIDE (genres)
genre_cols = [
"FPS","Platform","Adventure","RPG","Sports","Racing","Strategic","Simulation","MMO","Fighting",
    "Logic/puzzle","Horror","Educational","Rytmic/music","Survival","Open-world","Point and click adventure",
    "Tower defense","Battle royale","Sandbox","VR","Interactive films","Single player",
    "Multiplayer online","LAN","Multiplayer splitscreen","Cooperation"
]
df[genre_cols] = df[genre_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
genres = df[genre_cols]

# RIGHT SIDE 
#outcomes = df[["Mężczyzna", "Kobieta", "18-24", "25-32", "33+"]].copy()
outcomes = ['Bardzo rzadko', 'Często' ,'Sporadycznie', 'Bardzo często', 'Nigdy']
outcomes = df[outcomes].apply(pd.to_numeric, errors='coerce')

# Correlation matrix (genres vs outcomes)
corr = genres.apply(lambda col: outcomes.corrwith(col))

# Plot heatmap
plt.figure(figsize=(8, 12))
sns.heatmap(corr.T, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("How often do you turn to video games as a form of escape from your daily responsibilities?")
plt.show()


scaler = StandardScaler()
X = scaler.fit_transform(df[genre_cols])

# -----------------------------
# 3. Choose number of clusters (k)
# -----------------------------
k = 3  # You can change this to try k=3,4,5,...

kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# -----------------------------
# 4. See cluster profiles
# -----------------------------
print("\nCluster Profiles (mean values):")
print(df.groupby("Cluster")[genre_cols].mean())

# -----------------------------
# 5. Simple PCA plot
# -----------------------------

pca = PCA(n_components=2, random_state=42)
pts = pca.fit_transform(X)
print(f"Silhouette: {silhouette_score(X, df["Cluster"] ) }")
# Simplify column names
gender_col = "Please specify your gender. - Selected Choice"
age_col = "Please indicate your age (enter a two-digit number, such as 20)."

plt.figure(figsize=(8,7))

# gender → shape
shapes = {
    "Mężczyzna": "o",
    "Kobieta": "s",
    "Inna (jaka?)": "D"
}

# fallback for unknown labels
df[gender_col] = df[gender_col].replace(shapes.keys(), shapes.keys()).fillna("Other")

# normalize age → marker size
sizes = 50 + 4 * (df[age_col] - df[age_col].min())

for gender, marker in shapes.items():
    mask = df[gender_col] == gender
    plt.scatter(
        pts[mask, 0],
        pts[mask, 1],
        c=df.loc[mask, "Cluster"],
        cmap="tab10",
        s=sizes[mask],
        marker=marker,
        alpha=0.8,
        label=gender
    )

plt.title("Clusters by Genre Preference\nColor = Cluster | Shape = Gender | Size = Age")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Gender")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

