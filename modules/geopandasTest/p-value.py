import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import pandas as pd
import numpy as np


def export_to_csv(geo_df, output_file="output.csv"):
    """
    Exports the state names and calculated variable from a GeoDataFrame to a CSV file.

    Parameters:
        geo_df (GeoDataFrame): The GeoDataFrame containing state names and calculated values.
        output_file (str): The file path for the output CSV file (default is "output.csv").

    Returns:
        None
    """
    output_df = geo_df[["NAME", "New_Variable"]].rename(columns={"NAME": "State", "New_Variable": "Calculated_Value"})
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
data_path = "Updated_Data.csv"  # Adjust path if necessary
data = pd.read_csv(data_path)

# Print column names to ensure we use the correct ones
print("Available columns in the dataset:")
print(data.columns)

# Define the columns to use by their indices (0-based)
selected_columns_indices = [6, 7, 9, 10, 11]  # Replace with your desired indices
selected_columns = data.columns[selected_columns_indices]

# Define coefficients and p-values for the selected columns
coefficients = {
    "Median Temperature": 0.6820,
    "Urban Population %": -0.7327,
    "Median Income": 1.5787,
    "Population": 0.1982,
    "Cost of Living Index": -1.3283
}
p_values = {
    "Median Temperature": 0.111,
    "Urban Population %": 0.128,
    "Median Income": 0.010,
    "Population": 0.610,
    "Cost of Living Index": 0.005
}

# Ensure column names match coefficients and p_values
selected_columns_mapped = [col for col in selected_columns if col in coefficients]
if len(selected_columns_mapped) != len(selected_columns):
    print("Warning: Some selected columns do not match coefficient/p_value definitions. They will be skipped.")

# Normalize the selected columns
normalized_data = data[selected_columns_mapped].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Calculate the new variable using normalized values
data["New_Variable"] = sum(
    normalized_data[col] * coefficients[col] * (1 - p_values[col])
    for col in selected_columns_mapped
)

# Map the new variable to the GeoDataFrame using the state names
state_values = data.set_index("State")["New_Variable"].to_dict()
us_states["New_Variable"] = us_states["NAME"].map(state_values)

# Drop states without the new variable data (if any)
us_states = us_states.dropna(subset=["New_Variable"])

export_to_csv(us_states, output_file="calculated_values.csv")

# Calculate the range for color scale dynamically
min_new_variable = us_states["New_Variable"].min()
max_new_variable = us_states["New_Variable"].max()

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot state boundaries
us_states.boundary.plot(ax=ax, linewidth=1)

# Adjust color scale dynamically based on actual data range
us_states.plot(
    column="New_Variable",
    cmap="coolwarm",  # Reversed color map
    linewidth=0.8,
    edgecolor="black",
    legend=False,
    ax=ax,
    missing_kwds={"color": "lightgrey"},
    norm=plt.Normalize(vmin=min_new_variable, vmax=max_new_variable),
)

# Add state abbreviations with contrasting outline
for x, y, label in zip(us_states.geometry.centroid.x, us_states.geometry.centroid.y, us_states["STUSPS"]):
    ax.text(
        x, y, label, fontsize=8, ha='center', va='center',
        color="white", weight="bold",
        path_effects=[patheffects.withStroke(linewidth=2, foreground="black")]
    )

# Add title and color bar
plt.title("Calculated Variable by State - P-Value Weighted", fontsize=16)
plt.axis("off")

# Display a single color bar dynamically based on actual data range
sm = plt.cm.ScalarMappable(
    cmap="coolwarm", norm=plt.Normalize(vmin=min_new_variable, vmax=max_new_variable)
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Calculated Variable", fontsize=12)

plt.show()
