import pandas as pd
import numpy as np
from scipy.interpolate import griddata, Rbf
import matplotlib.pyplot as plt

# Load the DFOS strain data from the uploaded Excel files
vertical_df = pd.read_excel('Vertical.xlsx')
horizontal_df = pd.read_excel('Horizontal.xlsx')
diagonal_df = pd.read_excel('Diagonal.xlsx')

# Define the mesh grid for interpolation based on the structure dimensions
common_x = np.linspace(0, 0.8, 300)  # Higher resolution
common_y = np.linspace(0, 0.5, 300)
common_xi, common_yi = np.meshgrid(common_x, common_y)

# Function for interpolating with extrapolation
def interpolate_strain_with_extrapolation(df, strain_column, xi, yi):
    points = df[['X', 'Y']].values
    values = df[strain_column].values
    # Use Radial Basis Function (Rbf) for interpolation with extrapolation
    rbf = Rbf(points[:, 0], points[:, 1], values, function='linear')
    return rbf(xi, yi)

# Interpolate strain data for 200 kN load level
interpolated_horizontal = interpolate_strain_with_extrapolation(horizontal_df, 'Strain 200', common_xi, common_yi)
interpolated_vertical = interpolate_strain_with_extrapolation(vertical_df, 'Strain 200', common_xi, common_yi)
interpolated_diagonal = interpolate_strain_with_extrapolation(diagonal_df, 'Strain 200', common_xi, common_yi)

# Calculate Principal Strains (ε_max, ε_min) and Shear Strain (γ_max)
e_max = (interpolated_horizontal + interpolated_vertical) / 2 + np.sqrt(
    ((interpolated_horizontal - interpolated_vertical) / 2) ** 2 + interpolated_diagonal ** 2
)
e_min = (interpolated_horizontal + interpolated_vertical) / 2 - np.sqrt(
    ((interpolated_horizontal - interpolated_vertical) / 2) ** 2 + interpolated_diagonal ** 2
)
gamma_max = np.abs(interpolated_diagonal)

# Function to plot strain maps with smoothing and grid
def plot_strain_map(xi, yi, strain, title, filename):
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xi, yi, strain, levels=100, cmap='plasma', extend='both')
    plt.colorbar(contour, label=title)
    plt.title(title, fontsize=16)
    plt.xlabel('Length of the RC Wall [m]')
    plt.ylabel('Width of the RC Wall [m]')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.axis('equal')
    plt.savefig(filename, dpi=300)
    plt.show()

# Plot the strain maps with extrapolation
plot_strain_map(common_xi, common_yi, e_max, 'Principal Strain (ε_max) , Applied Load 200 kN', 'principal_strain_emax_200KN.png')
plot_strain_map(common_xi, common_yi, e_min, 'Principal Strain (ε_min) , Applied Load 200 kN', 'principal_strain_emin_200KN.png')
plot_strain_map(common_xi, common_yi, gamma_max, 'Shear Strain (γ_max) , Applied Load 200 kN', 'shear_strain_gamma_max_200KN.png')

# Export specific points with their computed strain values to an Excel file
specific_points = [
    [0, 0], [0.8, 0], [0.8, 0.5], [0, 0.5], [0, 0.03], [0.8, 0.03],
    [0, 0.14], [0.8, 0.14], [0, 0.25], [0.8, 0.25], [0, 0.36], [0.8, 0.36],
    [0, 0.47], [0.8, 0.47], [0.015, 0], [0.015, 0.5], [0.125, 0], [0.125, 0.5],
    [0.235, 0], [0.235, 0.5], [0.345, 0], [0.345, 0.5], [0.455, 0], [0.455, 0.5],
    [0.565, 0]
]

specific_points = np.array(specific_points)
specific_x = specific_points[:, 0]
specific_y = specific_points[:, 1]

# Interpolating strain data for specific points
e_max_specific = interpolate_strain_with_extrapolation(horizontal_df, 'Strain 200', specific_x, specific_y)
e_min_specific = interpolate_strain_with_extrapolation(vertical_df, 'Strain 200', specific_x, specific_y)
gamma_max_specific = interpolate_strain_with_extrapolation(diagonal_df, 'Strain 200', specific_x, specific_y)

# Creating DataFrame to save
output_df = pd.DataFrame({
    'X': specific_x,
    'Y': specific_y,
    'Principal Strain (ε_max)': e_max_specific,
    'Principal Strain (ε_min)': e_min_specific,
    'Shear Strain (γ_max)': gamma_max_specific
})

output_df.to_excel('strain_data_at_points_200KN.xlsx', index=False)
print("Excel file with strain data exported successfully.")
