import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("vgsales.csv")
df = df.dropna()

# 2. Correlation Analysis
corr_vars = ['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']
corr = df[corr_vars].corr()
print(corr)

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Regional and Global Sales")
plt.show()

# 3. Multiple Linear Regression (OLS)
# Rank ~ Year + Regional Sales + Global Sales
X_ols = df[['Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales']]
X_ols = sm.add_constant(X_ols)
y_ols = df['Rank']

ols_model = sm.OLS(y_ols, X_ols).fit()
print(ols_model.summary())

# 4. Binary Variable (Above / Below Average Rank)
df['AboveAvg'] = (df['Rank'] < df['Rank'].mean()).astype(int)

# 5. Binary Logit Regression
X_logit = df[['Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales']]
X_logit = sm.add_constant(X_logit)
y_logit = df['AboveAvg']

logit_model = sm.Logit(y_logit, X_logit).fit()
print(logit_model.summary())

# 6. Binary Probit Regression
probit_model = sm.Probit(y_logit, X_logit).fit()
print(probit_model.summary())

# 7. Random Forest (Feature Importance)
X = df[['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
y = df['AboveAvg']  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance_df)