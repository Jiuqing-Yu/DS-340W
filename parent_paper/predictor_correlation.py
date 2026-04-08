import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------- 1. Load & clean --------
df = pd.read_csv("vgsales.csv").dropna()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# -------- 2. Correlation heatmap --------
corr = df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -------- 3. Binary target --------
df["AboveAvg"] = (df["Rank"] < df["Rank"].mean()).astype(int)

# -------- 4. Train/test split for linear models --------
features_numeric = ['Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales']
X_train, X_test, y_train, y_test = train_test_split(df[features_numeric], df['AboveAvg'],
                                                    test_size=0.3, random_state=42)

# -------- 5. OLS / Logit / Probit --------
ols = smf.ols("Rank ~ Year + NA_Sales + EU_Sales + JP_Sales + Other_Sales",
              data=df).fit()
logit = smf.logit("AboveAvg ~ Year + NA_Sales + EU_Sales + JP_Sales + Other_Sales",
                  data=df).fit()
probit = smf.probit("AboveAvg ~ Year + NA_Sales + EU_Sales + JP_Sales + Other_Sales",
                    data=df).fit()
print(ols.summary())
print(logit.summary())
print(probit.summary())
# -------- 6. Convert OLS numeric → binary --------
threshold = df["Rank"].mean()
ols_pred_binary = (ols.predict(X_test) < threshold).astype(int)
logit_pred_binary = (logit.predict(X_test) >= 0.5).astype(int)
probit_pred_binary = (probit.predict(X_test) >= 0.5).astype(int)

# -------- 7. Random Forest with categorical --------
cat_cols = ["Platform","Genre","Publisher"]
num_cols = ["Year","NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]

X_rf = df[cat_cols + num_cols].copy()
y_rf = df["AboveAvg"].copy()

pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

rf = Pipeline([
    ("prep", pre),
    ("rf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
])

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42)
rf.fit(X_train_rf, y_train_rf)
rf_pred_binary = rf.predict(X_test_rf)

# -------- 8. Random Forest feature importance --------
feat = rf.named_steps["prep"].get_feature_names_out()
imp = rf.named_steps["rf"].feature_importances_
imp_df = pd.DataFrame({"feature": feat, "importance": imp}).sort_values("importance", ascending=False)

print("\nTop 25 one-hot features by RF importance:")
print(imp_df.head(25))

# Aggregate by original variable
def parent(name: str) -> str:
    if name.startswith("cat__Platform_"):  return "Platform"
    if name.startswith("cat__Genre_"):     return "Genre"
    if name.startswith("cat__Publisher_"): return "Publisher"
    if name.startswith("num__"):           return name.split("__", 1)[1]
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

# -------- 9. Metrics for all models --------
def compute_metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, rec, f1   # <-- RETURN metrics

# Now this will store the metrics
ols_metrics = compute_metrics(y_test, ols_pred_binary, "OLS")
logit_metrics = compute_metrics(y_test, logit_pred_binary, "Logistic Regression")
probit_metrics = compute_metrics(y_test, probit_pred_binary, "Probit")
rf_metrics = compute_metrics(y_test_rf, rf_pred_binary, "Random Forest")

# Create the comparison table
metrics_data = [
    {
        "Model": "OLS",
        "Accuracy": ols_metrics[0],
        "Precision": ols_metrics[1],
        "Recall": ols_metrics[2],
        "F1 Score": ols_metrics[3]
    },
    {
        "Model": "Logistic Regression",
        "Accuracy": logit_metrics[0],
        "Precision": logit_metrics[1],
        "Recall": logit_metrics[2],
        "F1 Score": logit_metrics[3]
    },
    {
        "Model": "Probit",
        "Accuracy": probit_metrics[0],
        "Precision": probit_metrics[1],
        "Recall": probit_metrics[2],
        "F1 Score": probit_metrics[3]
    },
    {
        "Model": "Random Forest",
        "Accuracy": rf_metrics[0],
        "Precision": rf_metrics[1],
        "Recall": rf_metrics[2],
        "F1 Score": rf_metrics[3]
    }
]

# Create DataFrame
comparison_df = pd.DataFrame(metrics_data)

# Display the table
print(comparison_df)