import nflreadpy as nfl
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
import polars as pl

# Load weekly player stats
player_stats = nfl.load_player_stats(range(2000, 2027))

# Pick your QBs (names must match the data's name column exactly)
qb_names = ["Tom Brady", "Peyton Manning", "Drew Brees",
            "Aaron Rodgers", "Patrick Mahomes"]

name_col = "player_display_name"   # if this errors, check player_stats.columns

qbs = player_stats.filter(
    (pl.col(name_col).is_in(qb_names))
    & (pl.col("position") == "QB")
    & (pl.col("season_type") == "REG")
)

# Candidate features; keep only those present in this schema
sum_feats = [
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered", "sack_yards_lost",
    "passing_air_yards", "passing_yards_after_catch", "passing_first_downs",
    "passing_epa", "carries", "rushing_yards", "rushing_tds",
]
mean_feats = ["passing_cpoe", "pacr"]   # rate-style stats: average, don't sum

sum_feats = [c for c in sum_feats if c in qbs.columns]
mean_feats = [c for c in mean_feats if c in qbs.columns]
print("Summed:", sum_feats)
print("Averaged:", mean_feats)

# One row per QB-season
seasons = (
    qbs.group_by([name_col, "season"])
    .agg(
        [pl.col(c).fill_null(0).sum() for c in sum_feats]
        + [pl.col(c).mean() for c in mean_feats]
    )
    .filter(pl.col("attempts") >= 100)      # drop cameo seasons
    .sort([name_col, "season"])
)
print(seasons.select([name_col, "season"]).group_by(name_col).len())

# Build matrix
features = sum_feats + mean_feats
X = seasons.select(features).fill_null(0).to_numpy().astype(float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print(X.shape)

# Center, standardize, SVD
Xc = X - X.mean(axis=0)
Xc = Xc / Xc.std(axis=0).clip(min=1e-12)
u, s, vt = LA.svd(Xc, full_matrices=False)

explained = s**2 / np.sum(s**2)
points = Xc @ vt[:2].T

# Plot: one color per QB, lines connect consecutive seasons
names = seasons[name_col].to_numpy()
years = seasons["season"].to_numpy()

plt.figure(figsize=(10, 7))
for name in qb_names:
    m = names == name
    if not m.any():
        print(f"No data found for {name}")
        continue
    plt.plot(points[m, 0], points[m, 1], "-o", label=name, alpha=0.7, markersize=5)
    for px, py, yr in zip(points[m, 0], points[m, 1], years[m]):
        plt.annotate(str(yr), (px, py), fontsize=7, xytext=(3, 3),
                     textcoords="offset points")

plt.xlabel(f"PC1 ({explained[0]:.1%})")
plt.ylabel(f"PC2 ({explained[1]:.1%})")
plt.title("QB seasons: SVD projection")
plt.legend()
plt.show()