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

