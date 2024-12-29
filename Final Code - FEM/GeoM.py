import matplotlib.pyplot as plt

# Structure dimensions
width = 800  # mm
height = 500  # mm

# Horizontal and vertical DFOS positions
horizontal_sensors = [50, 140, 250, 360, 470]  # y-coordinates in mm
vertical_sensors = [50, 150, 250, 350, 450, 550, 650, 750]  # x-coordinates in mm

# Corrected diagonal DFOS points
diagonal_start = [
    (100, 0), (200, 0), (300, 0), (400, 0), (500, 0),  # Starting along the bottom edge
    (600, 0), (700, 0), (800, 0), (800, 100), (800, 200), (800, 300), (800, 400)  # Starting along the right edge
]

diagonal_end = [
    (0, 100), (0, 200), (0, 300), (0, 400), (0, 500),  # Ending along the left edge
    (100, 500), (200, 500), (300, 500), (400, 500), (500, 500), (600, 500), (700, 500)  # Ending along the top edge
]

# Create the corrected plot
plt.figure(figsize=(12, 8))

# Draw structure boundary
plt.plot([0, width, width, 0, 0], [0, 0, height, height, 0], color='black', linewidth=1.5)

# Plot horizontal sensors
for y in horizontal_sensors:
    plt.plot([0, width], [y, y], color='blue', linestyle='-', label='Horizontal DFOS' if y == horizontal_sensors[0] else "")

# Plot vertical sensors
for x in vertical_sensors:
    plt.plot([x, x], [0, height], color='red', linestyle='-', label='Vertical DFOS' if x == vertical_sensors[0] else "")

# Plot diagonal sensors
for (x1, y1), (x2, y2) in zip(diagonal_start, diagonal_end):
    plt.plot([x1, x2], [y1, y2], color='green', linestyle='-', label='Diagonal DFOS' if (x1, y1) == diagonal_start[0] else "")

# Add annotations for horizontal sensors
for i, y in enumerate(horizontal_sensors, start=1):
    plt.text(-30, y, f"h{i}", fontsize=10, color='blue', va='center')

# Add annotations for vertical sensors
for i, x in enumerate(vertical_sensors, start=1):
    plt.text(x, -30, f"v{i}", fontsize=10, color='red', ha='center')

# Add annotations for diagonal sensors
for i, (start, end) in enumerate(zip(diagonal_start, diagonal_end), start=1):
    mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    plt.text(mid_x, mid_y, f"d{i}", fontsize=9, color='green', ha='center')


# Set labels, legend, and grid
plt.title("DFOS Distribution Over Structure", fontsize=14)
plt.xlabel("Width (mm)", fontsize=12)
plt.ylabel("Height (mm)", fontsize=12)
plt.legend(fontsize=8, loc='best')
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.8)

# Save and display the plot
plt.savefig("DFOS Distribution.png", dpi=300)
plt.show()
