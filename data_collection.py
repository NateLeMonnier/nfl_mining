import nflreadpy as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg as LA
import polars as pl
import polars.selectors as cs

# Import player data since 2000 to pandas
player_stats = nfl.load_player_stats(range(2000, 2026))
# players_stats = player_stats.to_pandas()
# player_df = pd.DataFrame(player_stats)

# Find number of data points and dimensions
# print(f"Number of Data Points and Dimensions: {player_df.shape}")

# Compute SVD of player data
player_data = player_stats.select(cs.numeric()).fill_null(0).to_numpy()
player_data = np.nan_to_num(player_data, nan=0.0, posinf=0.0, neginf=0.0)
print(player_data.shape)

# Define centering matrix
n = 24144
ones = np.ones((n, 1))
cn = np.identity(n) - (1 / n) * ones @ ones.T

# Center the data
playercent = cn @ player_data

# Compute SVD of player data
u, s, vt = LA.svd(playercent, full_matrices=False)
sigma = np.diag(s)

# Plot in 2 dimensions
rsv = vt[:2, :]

# Plot the points
points = playercent @ rsv.T
x = points[:, 0]
y = points[:, 1]
plt.scatter(x, y)
plt.show()