import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import pandas as pd
import numpy as np


def export_to_csv(geo_df, column_name="New_Variable", output_file="output.csv"):
    """
    Exports the state names and a calculated variable from a GeoDataFrame to a CSV file.

    Parameters:
        geo_df (GeoDataFrame): The GeoDataFrame containing state names and calculated values.
        column_name (str): The name of the column to export as the calculated value.
        output_file (str): The file path for the output CSV file (default is "output.csv").

    Returns:
        None
    """
    output_df = geo_df[["NAME", column_name]].rename(columns={"NAME": "State", column_name: "Calculated_Value"})
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
data_path = "Updated_Data.csv"
data = pd.read_csv(data_path)

# Define coefficients and p-values for all columns
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

# Normalize the selected columns
normalized_data = data[list(coefficients.keys())].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# Function to visualize with \( p \)-values less than 0.05 and \( p = 0 \)
def visualize_low_p_values():
    # Filter columns with p-values < 0.05
    low_p_columns = [col for col, p in p_values.items() if p < 0.05]

    # Recalculate the new variable assuming \( p = 0 \) for these columns
    data["New_Variable_LowP"] = sum(
        normalized_data[col] * coefficients[col] for col in low_p_columns
    )

    # Map the new variable to the GeoDataFrame
    state_values = data.set_index("State")["New_Variable_LowP"].to_dict()
    us_states["New_Variable_LowP"] = us_states["NAME"].map(state_values)

    # Drop states without the new variable data (if any)
    us_states_filtered = us_states.dropna(subset=["New_Variable_LowP"])

    export_to_csv(us_states_filtered, column_name="New_Variable_LowP", output_file="calculated_values.csv")

    # Calculate the range for color scale dynamically
    min_new_variable = us_states_filtered["New_Variable_LowP"].min()
    max_new_variable = us_states_filtered["New_Variable_LowP"].max()

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot state boundaries
    us_states_filtered.boundary.plot(ax=ax, linewidth=1)

    # Adjust color scale dynamically based on actual data range
    us_states_filtered.plot(
        column="New_Variable_LowP",
        cmap="coolwarm",
        linewidth=0.8,
        edgecolor="black",
        legend=False,
        ax=ax,
        missing_kwds={"color": "lightgrey"},
        norm=plt.Normalize(vmin=min_new_variable, vmax=max_new_variable),
    )

    # Add state abbreviations with contrasting outline
    for x, y, label in zip(us_states_filtered.geometry.centroid.x, us_states_filtered.geometry.centroid.y, us_states_filtered["STUSPS"]):
        ax.text(
            x, y, label, fontsize=8, ha='center', va='center',
            color="white", weight="bold",
            path_effects=[patheffects.withStroke(linewidth=2, foreground="black")]
        )

    # Add title and color bar
    plt.title("Statistically Significant Variables", fontsize=16)
    plt.axis("off")

    # Display a single color bar dynamically based on actual data range
    sm = plt.cm.ScalarMappable(
        cmap="coolwarm", norm=plt.Normalize(vmin=min_new_variable, vmax=max_new_variable)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Calculated Variable (p < 0.05)", fontsize=12)

    plt.show()


# Call the visualization function
visualize_low_p_values()
