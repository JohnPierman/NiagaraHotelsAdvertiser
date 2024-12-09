import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize


def export_to_csv(geo_df, output_file="output.csv"):
    """
    Exports the state names and calculated variable from a GeoDataFrame to a CSV file.

    Parameters:
        geo_df (GeoDataFrame): The GeoDataFrame containing state names and calculated values.
        output_file (str): The file path for the output CSV file (default is "output.csv").

    Returns:
        None
    """
    output_df = geo_df[["NAME", "Final_Score"]].rename(columns={"NAME": "State", "Final_Score": "Calculated_Value"})
    output_df.to_csv(output_file, index=False)
    print(f"Exported calculated values to {output_file}")


# Load the shapefile
shapefile_path = "cb_2022_us_state_20m.shp"
us_states = gpd.read_file(shapefile_path)

# Filter out Alaska, Hawaii, and DC
us_states = us_states[~us_states["NAME"].isin(["Alaska", "Hawaii", "District of Columbia"])]

# Re-project geometry to a projected CRS
us_states = us_states.to_crs(epsg=3857)

# Load the dataset
data_path = "table.csv"  # Replace with the correct path to the CSV file
data = pd.read_csv(data_path)

# Ensure the table uses "Final_Score" and "State"
if "Final_Score" not in data.columns or "State" not in data.columns:
    raise ValueError("The dataset must contain 'Final_Score' and 'State' columns.")

# Normalize the final score for consistent color mapping
data["Final_Score"] = (data["Final_Score"] - data["Final_Score"].min()) / (data["Final_Score"].max() - data["Final_Score"].min())

# Map the final score to the GeoDataFrame using the state names
state_values = data.set_index("State")["Final_Score"].to_dict()
us_states["Final_Score"] = us_states["NAME"].map(state_values)

# Drop states without the new variable data (if any)
us_states = us_states.dropna(subset=["Final_Score"])

# Export the data
export_to_csv(us_states, output_file="calculated_values.csv")

# Calculate the range for color scale dynamically
min_score = us_states["Final_Score"].min()
max_score = us_states["Final_Score"].max()

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot state boundaries
us_states.boundary.plot(ax=ax, linewidth=1)

# Adjust color scale dynamically based on actual data range
us_states.plot(
    column="Final_Score",
    cmap="coolwarm",  # Reversed color map
    linewidth=0.8,
    edgecolor="black",
    legend=False,
    ax=ax,
    missing_kwds={"color": "lightgrey"},
    norm=Normalize(vmin=min_score, vmax=max_score),
)

# Add state abbreviations with contrasting outline
for x, y, label in zip(us_states.geometry.centroid.x, us_states.geometry.centroid.y, us_states["STUSPS"]):
    ax.text(
        x, y, label, fontsize=8, ha='center', va='center',
        color="white", weight="bold",
        path_effects=[patheffects.withStroke(linewidth=2, foreground="black")]
    )

# Add title and color bar
plt.title("Market Penetration Score", fontsize=16)
plt.axis("off")

# Display a single color bar dynamically based on actual data range
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=Normalize(vmin=min_score, vmax=max_score))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Normalized Market Penetration Score", fontsize=12)

plt.show()
