import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.core as cfc
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define material properties
t = 0.1  # Thickness of the material (concrete)
v = 0.35  # Poisson's ratio
E_concrete = 32e9  # Young’s modulus for concrete (Pa)
E_steel = 200e9  # Young's modulus for steel reinforcement (Pa)

ptype = 1  # Plane stress condition
D_concrete = cfc.hooke(ptype, E_concrete, v)
D_steel = cfc.hooke(ptype, E_steel, v)

# Create geometry for the RC wall
g = cfg.Geometry()
g.point([0, 0])  # Bottom-left corner
g.point([0.8, 0])  # Bottom-right corner
g.point([0.8, 0.5])  # Top-right corner
g.point([0, 0.5])  # Top-left corner

# Define edges of the rectangle
for i in range(4):
    g.line([i, (i+1) % 4], marker=55)  # Mark concrete edges

# Add horizontal reinforcement bars
for y_pos in [0.03, 0.14, 0.25, 0.36, 0.47]:
    g.point([0, y_pos])
    g.point([0.8, y_pos])
    g.line([len(g.points) - 2, len(g.points) - 1], marker=66)

# Add vertical reinforcement bars
for x_pos in [0.015 + i*0.11 for i in range(8)]:
    g.point([x_pos, 0])
    g.point([x_pos, 0.5])
    g.line([len(g.points) - 2, len(g.points) - 1], marker=66)

# Create the concrete surface
g.surface([0, 1, 2, 3], marker=55)

# Mesh generation
mesh = cfm.GmshMeshGenerator(g)
mesh.el_size_factor = 0.05
mesh.el_type = 2
mesh.dofs_per_node = 2
coords, edof, dofs, bdofs, elementmarkers = mesh.create()

n_nodes = len(coords)

def plot_complete_mesh(coords, edof, reinf_horizontal, reinf_vertical, title="Complete Mesh Visualization", filename="complete_mesh.png"):
    """
    Visualize the complete mesh with nodes, reinforcements, and triangular elements.

    Parameters:
    - coords: Array of node coordinates.
    - edof: Element degree of freedom connectivity.
    - reinf_horizontal: List of Y positions for horizontal reinforcements.
    - reinf_vertical: List of X positions for vertical reinforcements.
    - title: Title for the plot.
    - filename: File to save the plot.
    """

    triangle_connectivity = (edof[:, [0, 2, 4]] // 2)  # Convert DOF indices to node indices

    # Create the plot
    plt.figure(figsize=(12, 10))

    # Plot the mesh (concrete elements)
    plt.triplot(coords[:, 0], coords[:, 1], triangle_connectivity, color='gray', label="Concrete Mesh")

    # Plot the nodes
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=15, label="Nodes")

    # Plot horizontal reinforcements
    for y in reinf_horizontal:
        plt.plot([0, 0.8], [y, y], 'b--', linewidth=1.5, label="Horizontal Reinforcement" if y == reinf_horizontal[0] else "")

    # Plot vertical reinforcements
    for x in reinf_vertical:
        plt.plot([x, x], [0, 0.5], 'g--', linewidth=1.5, label="Vertical Reinforcement" if x == reinf_vertical[0] else "")

    # Add labels, legend, and grid
    plt.title(title, fontsize=16)
    plt.xlabel("Length of the RC Wall (m)", fontsize=12)
    plt.ylabel("Width of the RC Wall (m)", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.axis('equal')
    plt.grid(True)

    # Save and display the plot
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Complete mesh visualization saved as {filename}.")

# Define reinforcement positions
reinf_horizontal = [0.03, 0.14, 0.25, 0.36, 0.47]  # Y positions of horizontal reinforcements
reinf_vertical = [0.015 + i * 0.11 for i in range(8)]  # X positions of vertical reinforcements

# Call the function to visualize the complete mesh
plot_complete_mesh(coords, edof, reinf_horizontal, reinf_vertical, title="Complete Mesh for RC Wall", filename="complete_mesh.png")


# Load strain data
strain_data_path = r"D:\MSc Thesis\Final PyPy\Data\LC 200.xlsx"

strain_df = pd.read_excel(strain_data_path)

strain_df.columns = strain_df.columns.str.strip()

# Check available columns for consistency
print("Available columns in strain_df:", strain_df.columns)

# Extract coordinates and strains for boundary condition nodes

bc_coords = strain_df[['X', 'Y']].values  # Coordinates of strain measurement points
horizontal_strains = strain_df.get('Strain X', np.nan).values  # Strain X (Horizontal)
vertical_strains = strain_df.get('Strain Y', np.nan).values    # Strain Y (Vertical)
diagonal_strains = strain_df.get('Strain XY', np.nan).values   # Strain XY (Shear)

# Map strain data to mesh nodes
mesh_kdtree = cKDTree(coords)  # KDTree for efficient neighbor searching
closest_indices = mesh_kdtree.query(bc_coords)[1]  # Find nearest mesh nodes for boundary conditions

# Initialize known strains (n_nodes x 3 array: Horizontal, Vertical, Diagonal)
known_strains = np.full((n_nodes, 3), np.nan)

# Populate known strains at the closest nodes
for i, node_index in enumerate(closest_indices):
    known_strains[node_index, 0] = horizontal_strains[i] if not np.isnan(horizontal_strains[i]) else np.nan
    known_strains[node_index, 1] = vertical_strains[i] if not np.isnan(vertical_strains[i]) else np.nan
    known_strains[node_index, 2] = diagonal_strains[i] if not np.isnan(diagonal_strains[i]) else np.nan

# Verify strain data mapped to mesh
print("Mapped known strains to mesh nodes.")

# Map strain data to mesh nodes
mesh_kdtree = cKDTree(coords)  # KDTree for efficient neighbor searching
closest_indices = mesh_kdtree.query(bc_coords)[1]  # Find nearest mesh nodes for boundary conditions

# Initialize known strains (n_nodes x 3 array: Horizontal, Vertical, Diagonal)
known_strains = np.full((n_nodes, 3), np.nan)

# Populate known strains at the closest nodes
for i, node_index in enumerate(closest_indices):
    known_strains[node_index, 0] = horizontal_strains[i] if not np.isnan(horizontal_strains[i]) else np.nan
    known_strains[node_index, 1] = vertical_strains[i] if not np.isnan(vertical_strains[i]) else np.nan
    known_strains[node_index, 2] = diagonal_strains[i] if not np.isnan(diagonal_strains[i]) else np.nan

# Split nodes into known and unknown strains
known_nodes = []
unknown_nodes = []

for i in range(n_nodes):
    if not np.isnan(known_strains[i]).all():  # At least one strain value is known
        known_nodes.append((i, coords[i, 0], coords[i, 1], *known_strains[i]))
    else:  # All strain values are NaN, meaning unknown
        unknown_nodes.append((i, coords[i, 0], coords[i, 1]))

# Convert to DataFrames for exporting
known_nodes_df = pd.DataFrame(known_nodes, columns=['Node Index', 'X', 'Y', 'Strain X', 'Strain Y', 'Strain XY'])
unknown_nodes_df = pd.DataFrame(unknown_nodes, columns=['Node Index', 'X', 'Y'])

# Export to Excel files
known_nodes_df.to_excel("mesh_nodes_with_known_strain.xlsx", index=False)
unknown_nodes_df.to_excel("mesh_nodes_with_unknown_strain.xlsx", index=False)

print("Mesh nodes with known strains exported to 'mesh_nodes_with_known_strain.xlsx'.")
print("Mesh nodes with unknown strains exported to 'mesh_nodes_with_unknown_strain.xlsx'.")

def plot_strain_data_types(coords, known_nodes, title="Strain Data Classification", filename="strain_data_classification.png"):
    """
    Visualize all nodes with known nodes classified by the type of strain directions.

    Parameters:
    - coords: Array of node coordinates.
    - known_nodes: List of nodes with assigned strain data.
    - title: Title for the plot.
    - filename: File to save the plot.
    """
    # Prepare data for classification
    all_directions = []  # Nodes with all three strains
    x_direction = []     # Nodes with X-direction strain only
    y_direction = []     # Nodes with Y-direction strain only
    diagonal_direction = []  # Nodes with diagonal strain only

    # Extract node indices of known nodes
    known_indices = [node[0] for node in known_nodes]

    for node in known_nodes:
        # Extract strains
        _, x, y, strain_x, strain_y, strain_xy = node
        num_known = sum(~np.isnan([strain_x, strain_y, strain_xy]))

        if num_known == 3:
            all_directions.append((x, y))
        elif num_known == 1:
            if not np.isnan(strain_x):
                x_direction.append((x, y))
            elif not np.isnan(strain_y):
                y_direction.append((x, y))
            elif not np.isnan(strain_xy):
                diagonal_direction.append((x, y))

    # Convert to numpy arrays for plotting
    all_directions = np.array(all_directions)
    x_direction = np.array(x_direction)
    y_direction = np.array(y_direction)
    diagonal_direction = np.array(diagonal_direction)

    plt.figure(figsize=(12, 10))

    # Plot all mesh nodes
    plt.scatter(coords[:, 0], coords[:, 1], c='gray', s=15, label="All Nodes", alpha=0.5)

    # Highlight known nodes
    if len(all_directions) > 0:
        plt.scatter(all_directions[:, 0], all_directions[:, 1], c='red', s=30, label="All Directional Strains")
    if len(x_direction) > 0:
        plt.scatter(x_direction[:, 0], x_direction[:, 1], c='orange', s=30, label="X-Directional Strain")
    if len(y_direction) > 0:
        plt.scatter(y_direction[:, 0], y_direction[:, 1], c='green', s=30, label="Y-Directional Strain")
    if len(diagonal_direction) > 0:
        plt.scatter(diagonal_direction[:, 0], diagonal_direction[:, 1], c='blue', s=30, label="XY-Directional Strain")

    # Add labels and legend
    plt.title(title, fontsize=16)
    plt.xlabel("Length of the RC Wall (m)", fontsize=12)
    plt.ylabel("Width of the RC Wall (m)", fontsize=12)
    plt.legend(fontsize=10)
    plt.axis('equal')
    plt.grid(True)

    # Save and show the plot
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Strain data classification visualization saved as {filename}.")

plot_strain_data_types(coords, known_nodes, title="Strain Data Classification", filename="strain_data_classification.png")

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import gmres

# Calculate the average distance to the nearest neighbor
distances, _ = mesh_kdtree.query(coords, k=2)  # k=2 to get the nearest neighbor (excluding the node itself)
avg_distance = np.mean(distances[:, 1])  # Use the second column (nearest neighbor distance)

dynamic_radius = avg_distance * 3.0

# Collect rows, columns, and values for sparse matrix assembly
rows = []
cols = []
values = []
b = np.zeros(3 * n_nodes)  # Right-hand side vector

for i in range(n_nodes):
    for j in range(3):  # For each strain component (H, V, D)
        if not np.isnan(known_strains[i, j]):
            rows.append(3 * i + j)
            cols.append(3 * i + j)
            values.append(1)
            b[3 * i + j] = known_strains[i, j]
        else:
            neighbors = mesh_kdtree.query_ball_point(coords[i], dynamic_radius)
            num_neighbors = len(neighbors)
            if num_neighbors > 1:
                rows.append(3 * i + j)
                cols.append(3 * i + j)
                values.append(num_neighbors)
                for neighbor in neighbors:
                    if neighbor != i:
                        rows.append(3 * i + j)
                        cols.append(3 * neighbor + j)
                        values.append(-1)

# Fix a boundary condition
rows.append(0)
cols.append(0)
values.append(1)
b[0] = 0

# Assemble sparse matrix
A = coo_matrix((values, (rows, cols)), shape=(3 * n_nodes, 3 * n_nodes))

# Solve the linear system using GMRES
x, exit_code = gmres(A, b, atol=1e-5, maxiter=10000)

if exit_code == 0:
    print("GMRES Solver converged!")
else:
    print(f"GMRES Solver did not converge after {10000} iterations.")

# Reshape solution into strains
optimized_strains = x.reshape((n_nodes, 3))

# visualize strain maps
strain_xx, strain_yy, strain_xy = optimized_strains.T

grid_x, grid_y = np.mgrid[
    coords[:, 0].min():coords[:, 0].max():200j,
    coords[:, 1].min():coords[:, 1].max():200j
]


def plot_contour_with_grids(grid_x, grid_y, grid_z, title, filename):
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='viridis')  # Smooth gradient
    plt.colorbar(contour, label=title)
    plt.title(title, fontsize=16)
    plt.xlabel('Length of the RC Wall (m)', fontsize=12)
    plt.ylabel('Width of the RC Wall (m)', fontsize=12)

    # Add grid lines for better grid-to-grid comparison
    plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.axis('equal')  # Ensure uniform scaling
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"{title} saved as {filename}")

# Generating strain maps
grid_z_xx = griddata(coords, strain_xx, (grid_x, grid_y), method='cubic')
grid_z_yy = griddata(coords, strain_yy, (grid_x, grid_y), method='cubic')
grid_z_xy = griddata(coords, strain_xy, (grid_x, grid_y), method='cubic')

# Plotting with grids
plot_contour_with_grids(grid_x, grid_y, grid_z_xx, 'Strain Map_200KN (Horizontal)', 'strain_map_horizontal_200KN.png')
plot_contour_with_grids(grid_x, grid_y, grid_z_yy, 'Strain Map_200KN (Vertical)', 'strain_map_vertical_200KN.png')
plot_contour_with_grids(grid_x, grid_y, grid_z_xy, 'Strain Map_200KN (Diagonal)', 'strain_map_diagonal_200KN.png')

# Calculate principal strains and maximum shear strain
epsilon_x = optimized_strains[:, 0]  # Horizontal strain
epsilon_y = optimized_strains[:, 1]  # Vertical strain
gamma_xy = optimized_strains[:, 2]  # Shear strain

# Principal strains
mean_strain = (epsilon_x + epsilon_y) / 2
diff_strain = (epsilon_x - epsilon_y) / 2
shear_component = gamma_xy / 2

Emax = mean_strain + np.sqrt(diff_strain**2 + shear_component**2)
Emin = mean_strain - np.sqrt(diff_strain**2 + shear_component**2)

# Maximum shear strain
Ymax = np.sqrt((epsilon_x - epsilon_y)**2 + gamma_xy**2)


grid_x, grid_y = np.mgrid[
    coords[:, 0].min():coords[:, 0].max():200j,
    coords[:, 1].min():coords[:, 1].max():200j
]


# Generating continuous maps

grid_Emax = griddata(coords, Emax, (grid_x, grid_y), method='cubic')
grid_Emin = griddata(coords, Emin, (grid_x, grid_y), method='cubic')
grid_Ymax = griddata(coords, Ymax, (grid_x, grid_y), method='cubic')

# Combine results into a DataFrame
strain_results = pd.DataFrame({
    'Node Index': np.arange(len(coords)),
    'X': coords[:, 0],
    'Y': coords[:, 1],
    'Emax': Emax,
    'Emin': Emin,
    'Ymax': Ymax
})

# Export to Excel
strain_results.to_excel("strain_results_principal_shear.xlsx", index=False)
print("Strain results saved as 'strain_results_principal_shear.xlsx'.")


def plot_map_with_grids(grid_x, grid_y, grid_z, title, colorbar_label, filename, cmap, vmin=None, vmax=None):
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=200, cmap=cmap, vmin=vmin, vmax=vmax)  # Smooth gradient
    plt.colorbar(contour, label=colorbar_label)
    plt.title(title, fontsize=16)
    plt.xlabel('Length of the RC Wall (m)', fontsize=12)
    plt.ylabel('Width of the RC Wall (m)', fontsize=12)

    # Add grid lines for better grid-to-grid comparison
    plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.axis('equal')  # Ensure uniform scaling
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"{title} saved as {filename}")


# Principal Strain (Emax): Sequential
plot_map_with_grids(grid_x, grid_y, grid_Emax, 'Principal Strain (ε_max) , Applied Load 200KN ', 'ε_max [μɛ]',
                    'principal_strain_emax_200KN.png', cmap='plasma')

# Principal Strain (Emin): Diverging
plot_map_with_grids(grid_x, grid_y, grid_Emin, 'Principal Strain (ε_min) , Applied Load 200KN ', 'ε_min [μɛ]',
                    'principal_strain_emin_200KN.png', cmap='plasma', vmin=-150, vmax=50)

# Shear Strain (Ymax): Sequential
plot_map_with_grids(grid_x, grid_y, grid_Ymax, 'Shear Strain (γ_max) , Applied Load 200KN ', 'γ_max [μɛ]',
                    'shear_strain_ymax_200KN.png', cmap='plasma')

# Calculate γmax (Maximum Shear Strain)
gamma_max = np.sqrt(2) * np.sqrt((strain_xx - strain_xy)**2 + (strain_xy - strain_yy)**2)

# Calculate Principal Strains
epsilon_max = 0.5 * (strain_xx + strain_yy + gamma_max)
epsilon_min = 0.5 * (strain_xx + strain_yy - gamma_max)

# Calculate Principal Angle (in radians)
theta = 0.5 * np.arctan2((strain_xx + strain_yy - 2 * strain_xy), (strain_xx - strain_yy))

# Convert Principal Angle to Degrees
theta_deg = np.degrees(theta)

# Combine results for all nodes
all_strain_results = pd.DataFrame({
    'Node Index': np.arange(len(coords)),
    'X Coordinate': coords[:, 0],
    'Y Coordinate': coords[:, 1],
    'Principal Strain (ε_max)': epsilon_max,
    'Principal Strain (ε_min)': epsilon_min,
    'Maximum Shear Strain (γ_max)': gamma_max,
    'Principal Angle (θ in degrees)': theta_deg
})

# Export to Excel
output_file = "all_nodes_strain_results.xlsx"
all_strain_results.to_excel(output_file, index=False)
print(f"All node strain results saved to '{output_file}'.")



