import nflreadpy as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import player data since 2000 to pandas
player_stats = nfl.load_player_stats([2000, 2026])
players_stats = player_stats.to_pandas()
player_df = pd.DataFrame(player_stats)

# Find number of data points and dimensions
print(f"Number of Data Points and Dimensions: {player_df.shape}")

