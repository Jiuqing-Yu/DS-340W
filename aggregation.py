
import pandas as pd
import re
import ast

# -------------------------
# Load files
# -------------------------
steam = pd.read_csv("steam.csv")                     # expects columns: name, genres
vpd   = pd.read_csv("Valve_Player_Data.csv")         # expects column: Game_Name

# -------------------------
# Normalize names for matching
# -------------------------
def norm(t):
    if pd.isna(t):
        return ""
    t = str(t).lower()
    t = re.sub(r"[®™©]", "", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

steam["norm_name"] = steam["name"].apply(norm)
vpd["norm_name"]   = vpd["Game_Name"].apply(norm)

# -------------------------
# Parse genres from steam.csv into Python lists
# -------------------------
def parse_genres(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()

    # Try to parse list-like: ["Action","RPG"]
    if s.startswith("[") and s.endswith("]"):
        try:
            vals = ast.literal_eval(s)
            if isinstance(vals, (list, tuple)):
                return [str(i).strip() for i in vals if str(i).strip()]
        except Exception:
            pass

    # Fallback: split by comma/semicolon
    return [p.strip() for p in re.split(r"[;,]", s) if p.strip()]

steam["genres_list"] = steam["genres"].apply(parse_genres)

# -------------------------
# Merge by normalized name
# -------------------------
merged = vpd.merge(
    steam[["norm_name", "genres_list"]],
    on="norm_name",
    how="left"
)

# -------------------------
# Ensure every row has a list (avoid fillna([]))
# -------------------------
def ensure_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x) or x is None:
        return []
    # If it's a string like "Action, RPG", parse it
    return parse_genres(x)

merged["genres_list"] = merged["genres_list"].apply(ensure_list)

# -------------------------
# One-hot encode genres
# -------------------------
# Collect all unique genres safely
all_genres = sorted({g for lst in merged["genres_list"] for g in (lst or [])})

# Create binary columns
for g in all_genres:
    merged[f"genre_{g}"] = merged["genres_list"].apply(lambda lst: 1 if g in lst else 0)

# -------------------------
# Save result
# -------------------------
merged.to_csv("Valve_Player_Data_with_genres.csv", index=False)
print("Done! Created", len(all_genres), "genre columns → Valve_Player_Data_with_genres.csv")


# After merged["genres_list"] is created
empty_genres = merged[merged["genres_list"].apply(len) == 0]

print("\nGames with empty genres_list:")
for name in empty_genres["Game_Name"]:
    print(name)
