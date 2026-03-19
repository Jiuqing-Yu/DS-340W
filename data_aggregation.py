
import pandas as pd
df = pd.read_csv("vgsales.csv")

# Drop missing critical values
df = df.dropna(subset=["Publisher", "Year", "Global_Sales"])
df["Year"] = df["Year"].astype(int)

# For each publisher-year, pick the game with MAX Global_Sales
top_sale = (
    df.loc[df.groupby(["Publisher", "Year"])["Global_Sales"].idxmax()]
      .sort_values(["Publisher", "Year"])
      .reset_index(drop=True)
)

# Save result
top_sale.to_csv("publisher_year_top_game.csv", index=False)

print(top_sale.head())
