"""
MSDS670 Final Project - Spotify Top Songs 2010-2019
Data Visualizations

"""

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------------------
CSV_PATH = "top50MusicFrom2010-2019.csv"

df = pd.read_csv(CSV_PATH)


df = df.rename(columns={
    "the genre of the track": "genre",
    "Beats.Per.Minute -The tempo of the song": "bpm",
    "Energy- The energy of a song - the higher the value, the more energtic": "energy",
    "Danceability - The higher the value, the easier it is to dance to this song": "danceability",
    "Loudness/dB - The higher the value, the louder the song": "loudness",
    "Liveness - The higher the value, the more likely the song is a live recording": "liveness",
    "Valence - The higher the value, the more positive mood for the song": "valence",
    "Length - The duration of the song": "length",
    "Acousticness - The higher the value the more acoustic the song is": "acousticness",
    "Speechiness - The higher the value the more spoken word the song contains": "speechiness",
    "Popularity- The higher the value the more popular the song is": "popularity",
})

# Basic cleaning: drop rows missing year or genre, coerce numerics.
df = df.dropna(subset=["year", "genre"])
df["year"] = df["year"].astype(int)
for col in ["energy", "danceability", "loudness", "popularity"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df[df["genre"].astype(str).str.strip() != ""]

# 2. LINE PLOT - time series of average energy & danceability by year
# ---------------------------------------------------------------------------
yearly = df.groupby("year")[["energy", "danceability"]].mean().round(1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(yearly.index, yearly["energy"], marker="o", label="Energy")
ax.plot(yearly.index, yearly["danceability"], marker="s", label="Danceability")

for col, off in [("energy", 8), ("danceability", -14)]:
    for x, y in zip(yearly.index, yearly[col]):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, off), ha="center", fontsize=8)

ax.set_title("Average Energy and Danceability of Top Spotify Songs, 2010-2019")
ax.set_xlabel("Year")
ax.set_ylabel("Audio Feature Score (0-100 scale)")
ax.set_xticks(yearly.index)
ax.legend()
ax.grid(False)
fig.tight_layout()
fig.savefig("plot_line_energy_danceability.png", dpi=200)


# 3. HORIZONTAL BAR PLOT - top 10 genres by song count, sorted
# ---------------------------------------------------------------------------
genre_counts = df["genre"].value_counts().head(10).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(genre_counts.index, genre_counts.values)

for y, v in enumerate(genre_counts.values):
    ax.annotate(f"{v:.0f}", (v, y), textcoords="offset points",
                xytext=(4, 0), va="center", fontsize=9)

ax.set_title("Top 10 Genres Among Spotify's Most Popular Songs, 2010-2019")
ax.set_xlabel("Number of Songs (count)")
ax.set_ylabel("Genre")
ax.grid(False)
fig.tight_layout()
fig.savefig("plot_bar_genres.png", dpi=200)


# 4. SCATTER PLOT - energy vs. loudness
# ---------------------------------------------------------------------------
# A handful of records carry impossible loudness values (e.g. one song listed
# at -60 dB with energy 0). These are data-entry errors that distort the axis.
scatter_df = df[df["loudness"] >= -12]

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(scatter_df["loudness"], scatter_df["energy"],
           alpha=0.5, edgecolors="none")

ax.set_title("Relationship Between Loudness and Energy in Top Songs, 2010-2019")
ax.set_xlabel("Loudness (decibels, dB)")
ax.set_ylabel("Energy Score (0-100 scale)")
ax.set_xlim(-12, 0)
ax.set_ylim(0, 100)
ax.grid(False)
fig.tight_layout()
fig.savefig("plot_scatter_energy_loudness.png", dpi=200)

print("Saved three PNGs.")
print("Songs analyzed:", len(df))
print("Correlation, all data (loudness vs energy):",
      round(df["loudness"].corr(df["energy"]), 2))
print("Correlation, outliers removed:",
      round(scatter_df["loudness"].corr(scatter_df["energy"]), 2))
print("Top genre:", df["genre"].value_counts().idxmax(),
      "with", df["genre"].value_counts().max(), "songs")
