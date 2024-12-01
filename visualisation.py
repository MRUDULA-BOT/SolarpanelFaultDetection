import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Helper function to enhance contrast
def enhance_contrast(data):
    data = np.clip(data, np.percentile(data, 5), np.percentile(data, 95))  # Clip extremes
    return (data - data.min()) / (data.max() - data.min())  # Normalize to [0, 1]

# Helper function for 2D visualization
def visualize_2d(data, title, cmap="viridis"):
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap=cmap, aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()

# Helper function for 3D visualization
def visualize_3d(data, title, colormap="viridis"):
    h, w = data.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, data, cmap=colormap, edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.set_zlabel("Pixel Intensity")
    plt.show()

# Load your data (replace paths with your file paths)
X_test = np.load(r'C:\Users\khv21\SolarPanels\mridula\mridula\X_test_n12c.npy')
y_test = np.load(r'C:\Users\khv21\SolarPanels\mridula\mridula\y_test_n12c.npy')

# Calculate averages
all_average = X_test.mean(axis=0)[:, :, 0]  # Average of all samples
cracked_indices = y_test.sum(axis=1) > 0
nice_indices = y_test.sum(axis=1) == 0
cracked_average = X_test[cracked_indices].mean(axis=0)[:, :, 0] if np.any(cracked_indices) else None
nice_average = X_test[nice_indices].mean(axis=0)[:, :, 0] if np.any(nice_indices) else None

# Enhance contrast for visualization
all_average = enhance_contrast(all_average)
if cracked_average is not None:
    cracked_average = enhance_contrast(cracked_average)
if nice_average is not None:
    nice_average = enhance_contrast(nice_average)

# Visualize averages
visualize_2d(all_average, "Average of All Samples")
visualize_3d(all_average, "3D Average of All Samples")

if cracked_average is not None:
    visualize_2d(cracked_average, "Average of Cracked Panels")
    visualize_3d(cracked_average, "3D Average of Cracked Panels")

if nice_average is not None:
    visualize_2d(nice_average, "Average of Nice Panels")
    visualize_3d(nice_average, "3D Average of Nice Panels")

# Visualize the difference between cracked and nice panels
if cracked_average is not None and nice_average is not None:
    difference_map = cracked_average - nice_average
    visualize_2d(difference_map, "Difference Between Cracked and Nice Panels", cmap="coolwarm")
    visualize_3d(difference_map, "3D Difference Between Cracked and Nice Panels", colormap="coolwarm")
