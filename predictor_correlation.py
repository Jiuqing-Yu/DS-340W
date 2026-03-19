
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------- 1. Load & clean --------
df = pd.read_csv("vgsales.csv").dropna()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# -------- 2. Correlation (article style) --------
corr = df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

# -------- 3. OLS / Logit / Probit (numeric only) --------
ols = smf.ols("Rank ~ Year + NA_Sales + EU_Sales + JP_Sales + Other_Sales",
              data=df).fit()
print(ols.summary())

df["AboveAvg"] = (df["Rank"] < df["Rank"].mean()).astype(int)

logit = smf.logit("AboveAvg ~ Year + NA_Sales + EU_Sales + JP_Sales + Other_Sales",
                  data=df).fit()
print(logit.summary())

probit = smf.probit("AboveAvg ~ Year + NA_Sales + EU_Sales + JP_Sales + Other_Sales",
                    data=df).fit()
print(probit.summary())


# -------- 4) Random Forest feature importance (includes categorical factors) --------
cat_cols = ["Platform","Genre","Publisher"]
num_cols = ["Year","NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]

X = df[cat_cols + num_cols].copy()
y = df["AboveAvg"].copy()

pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop"
)

rf = Pipeline([
    ("prep", pre),
    ("rf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
rf.fit(X_train, y_train)

feat = rf.named_steps["prep"].get_feature_names_out()
imp = rf.named_steps["rf"].feature_importances_
imp_df = pd.DataFrame({"feature": feat, "importance": imp}).sort_values("importance", ascending=False)

print("\nTop 25 one-hot features by RF importance:")
print(imp_df.head(25))

def parent(name: str) -> str:
    if name.startswith("cat__Platform_"):  return "Platform"
    if name.startswith("cat__Genre_"):     return "Genre"
    if name.startswith("cat__Publisher_"): return "Publisher"
    if name.startswith("num__"):           return name.split("__", 1)[1]  # e.g., 'Global_Sales'
    return name

agg = imp_df.copy()
agg["variable"] = agg["feature"].apply(parent)
agg = agg.groupby("variable", as_index=False)["importance"].sum().sort_values("importance", ascending=False)

print("\nAggregated RF importance by original variable:")
print(agg)

sns.barplot(data=agg, x="importance", y="variable", orient="h")
plt.title("Random Forest Feature Importance (Aggregated)")
plt.tight_layout()
plt.show()

