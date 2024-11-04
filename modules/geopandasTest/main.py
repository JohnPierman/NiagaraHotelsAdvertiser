import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Load the shapefile
shapefile_path = "cb_2022_us_state_20m.shp"
us_states = gpd.read_file(shapefile_path)

# Filter out Alaska, Hawaii, and DC
us_states = us_states[~us_states["NAME"].isin(["Alaska", "Hawaii", "District of Columbia"])]

# Population density data
population_density = {
    "Wyoming": 5.9, "Montana": 7.4, "North Dakota": 11.3, "South Dakota": 11.7,
    "New Mexico": 17.4, "Idaho": 24, "Nebraska": 25.5, "Nevada": 28.3, "Kansas": 35.9,
    "Utah": 39.8, "Oregon": 44.1, "Maine": 44.1, "Oklahoma": 57.7, "Arkansas": 57.8,
    "Mississippi": 63.1, "West Virginia": 74.5, "Iowa": 56, "Minnesota": 71.7,
    "Kentucky": 113.4, "South Carolina": 170.0, "Louisiana": 106.9, "Alabama": 99.0,
    "Wisconsin": 108.5, "Michigan": 177.4, "Indiana": 189.2, "Missouri": 89.3,
    "North Carolina": 214.3, "Georgia": 185.0, "Virginia": 218.0, "Tennessee": 167.7,
    "Texas": 111.3, "Ohio": 288.2, "Pennsylvania": 290.1, "Illinois": 230.5,
    "New York": 427.9, "Florida": 399.4, "Massachusetts": 896.7, "Rhode Island": 1050.1,
    "New Jersey": 1252.4
}

# Map population density to states
us_states["density"] = us_states["NAME"].map(population_density)

# Drop states without density data (if any)
us_states = us_states.dropna(subset=["density"])

# Calculate the median density for color scale centering
median_density = np.median(list(population_density.values()))

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
us_states.boundary.plot(ax=ax, linewidth=1)

# Adjust color scale to be centered around the median
us_states.plot(column="density", cmap="coolwarm", linewidth=0.8, edgecolor="black", legend=False, ax=ax,
               missing_kwds={"color": "lightgrey"},
               norm=plt.Normalize(vmin=median_density - 300, vmax=median_density + 300))

# Add title and color bar
plt.title("Population Density of Contiguous U.S. States (people per sq mile)", fontsize=16)
plt.axis("off")

# Display a single color bar centered on median
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=median_density - 300, vmax=median_density + 300))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Population Density (people per sq mile)", fontsize=12)

plt.show()
