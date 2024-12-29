import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load strain data
data_path = r"D:\MSc Thesis\Final PyPy\Data\LC 125.xlsx"
strain_data = pd.read_excel(data_path)

# Clean column names
strain_data.columns = strain_data.columns.str.strip()

# Extract coordinates and strain components
x_coords = strain_data['X'].values
y_coords = strain_data['Y'].values
strain_xx = strain_data['Strain X'].values
strain_yy = strain_data['Strain Y'].values
strain_xy = strain_data['Strain XY'].values

# Replace missing values with mean
strain_xx = np.nan_to_num(strain_xx, nan=np.nanmean(strain_xx))
strain_yy = np.nan_to_num(strain_yy, nan=np.nanmean(strain_yy))
strain_xy = np.nan_to_num(strain_xy, nan=np.nanmean(strain_xy))

# Define grid for interpolation
grid_x, grid_y = np.mgrid[
    x_coords.min():x_coords.max():200j,
    y_coords.min():y_coords.max():200j
]

# Perform interpolation
grid_strain_xx = griddata((x_coords, y_coords), strain_xx, (grid_x, grid_y), method='linear')
grid_strain_yy = griddata((x_coords, y_coords), strain_yy, (grid_x, grid_y), method='linear')
grid_strain_xy = griddata((x_coords, y_coords), strain_xy, (grid_x, grid_y), method='linear')

# Fill remaining NaN values
grid_strain_xx = np.nan_to_num(grid_strain_xx, nan=np.nanmean(grid_strain_xx))
grid_strain_yy = np.nan_to_num(grid_strain_yy, nan=np.nanmean(grid_strain_yy))
grid_strain_xy = np.nan_to_num(grid_strain_xy, nan=np.nanmean(grid_strain_xy))

# Plotting function
def plot_interpolated(grid_x, grid_y, grid_z, title, filename):
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='viridis')
    plt.colorbar(contour, label=title)
    plt.title(title, fontsize=14)
    plt.xlabel('length of the RC Wall (m)')
    plt.ylabel('Width of the RC Wall (m)')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axis('equal')
    plt.savefig(filename, dpi=300)
    plt.show()

# Visualize interpolated strain maps
plot_interpolated(grid_x, grid_y, grid_strain_xx, "Interpolated Strain , Applied Load 125KN", "strain_xx_125.png")
plot_interpolated(grid_x, grid_y, grid_strain_yy, "Interpolated Strain , Applied Load 125KN", "strain_yy_125.png")
plot_interpolated(grid_x, grid_y, grid_strain_xy, "Interpolated Strain , Applied Load 125KN", "strain_xy_125.png")

# Calculate components for principal strain
mean_strain = (grid_strain_xx + grid_strain_yy) / 2  # Mean strain
diff_strain = (grid_strain_xx - grid_strain_yy) / 2  # Difference strain
shear_component = grid_strain_xy / 2                # Shear component

# Calculate principal strains (εmax, εmin)
epsilon_max = mean_strain + np.sqrt(diff_strain**2 + shear_component**2)  # Principal max strain
epsilon_min = mean_strain - np.sqrt(diff_strain**2 + shear_component**2)  # Principal min strain

# Calculate maximum shear strain (γmax)
gamma_max = np.sqrt((grid_strain_xx - grid_strain_yy)**2 + grid_strain_xy**2)  # Max shear strain

# Plotting function for principal strains and shear strain
def plot_principal_map(grid_x, grid_y, grid_z, title, filename):
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='plasma')
    plt.colorbar(contour, label=title)
    plt.title(title, fontsize=14)
    plt.xlabel('length of the RC Wall (m)')
    plt.ylabel('Width of the RC Wall (m)')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axis('equal')
    plt.savefig(filename, dpi=300)
    plt.show()

# Plot Principal Strains and Shear Strain
plot_principal_map(grid_x, grid_y, epsilon_max, "Principal Strain (ε_max) , Applied Load 125KN", "principal_strain_epsilon_max_125KN.png")
plot_principal_map(grid_x, grid_y, epsilon_min, "Principal Strain (ε_min) ,  Applied Load 125KN", "principal_strain_epsilon_min_125KN.png")
plot_principal_map(grid_x, grid_y, gamma_max, "Shear Strain (γ_max) , Applied Load 125KN", "shear_strain_gamma_max_125KN.png")

# Define specific points to export
specific_points = [
    [0, 0], [0.8, 0], [0.8, 0.5], [0, 0.5], [0, 0.03], [0.8, 0.03],
    [0, 0.14], [0.8, 0.14], [0, 0.25], [0.8, 0.25], [0, 0.36], [0.8, 0.36],
    [0, 0.47], [0.8, 0.47], [0.015, 0], [0.015, 0.5], [0.125, 0], [0.125, 0.5],
    [0.235, 0], [0.235, 0.5], [0.345, 0], [0.345, 0.5], [0.455, 0], [0.455, 0.5],
    [0.565, 0]
]

specific_points = np.array(specific_points)

# Interpolate strain values at specific points
epsilon_max_values = griddata((grid_x.flatten(), grid_y.flatten()), epsilon_max.flatten(), (specific_points[:, 0], specific_points[:, 1]), method='linear')
epsilon_min_values = griddata((grid_x.flatten(), grid_y.flatten()), epsilon_min.flatten(), (specific_points[:, 0], specific_points[:, 1]), method='linear')
gamma_max_values = griddata((grid_x.flatten(), grid_y.flatten()), gamma_max.flatten(), (specific_points[:, 0], specific_points[:, 1]), method='linear')

# Prepare DataFrame to export
output_df = pd.DataFrame({
    'X': specific_points[:, 0],
    'Y': specific_points[:, 1],
    'Principal Strain (ε_max)': epsilon_max_values,
    'Principal Strain (ε_min)': epsilon_min_values,
    'Shear Strain (γ_max)': gamma_max_values
})

# Save to Excel
output_file = "strain_data_at_specific_points_125KN.xlsx"
output_df.to_excel(output_file, index=False)
print(f"Excel file saved at {output_file}")