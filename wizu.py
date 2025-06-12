import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

def load_data_points(filename):
    """Loads data points from a text file."""
    print(f"Loading data points from file: {filename}...")
    start_time = time.time()
    try:
        data = np.loadtxt(filename, dtype=np.float32)
        end_time = time.time()
        print(f"Loaded {data.shape[0]} data points with {data.shape[1]} dimensions in {end_time - start_time:.2f} seconds.")
        return data
    except FileNotFoundError:
        print(f"Error: Data file '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data file '{filename}': {e}")
        return None

def load_centroids(filename, type_name="centroids"):
    """Loads centroids from a text file into a NumPy array."""
    print(f"Loading {type_name} from file: {filename}...")
    start_time = time.time()
    try:
        data = np.loadtxt(filename, dtype=np.float32)
        end_time = time.time()
        print(f"Loaded {data.shape[0]} {type_name} in {end_time - start_time:.2f} seconds.")
        # Ensure centroids are 2D, even if there's only one
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading file '{filename}': {e}")
        return None

def load_assignments(filename, type_name="assignments"):
    """Loads point assignments from a text file."""
    print(f"Loading {type_name} from file: {filename}...")
    start_time = time.time()
    try:
        data = np.loadtxt(filename, dtype=np.int32)
        end_time = time.time()
        print(f"Loaded {data.shape[0]} {type_name} in {end_time - start_time:.2f} seconds.")
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading file '{filename}': {e}")
        return None


## Plotting Function (using pre-calculated assignments)

def plot_kmeans_results(data_points_file, cpu_centroids_file, gpu_centroids_file,
                        cpu_assignments_file, gpu_assignments_file,
                        output_filename='kmeans_comparison_plot.png'):
    """
    Loads data points, centroids, and pre-calculated assignments,
    then generates plots for CPU and GPU results, saving them to a file.
    Supports only 2-dimensional data.
    """
    print("\n--- Starting plot generation ---")
    start_total_plot_time = time.time()

    data_points = load_data_points(data_points_file)
    cpu_centroids = load_centroids(cpu_centroids_file, "CPU centroids")
    gpu_centroids = load_centroids(gpu_centroids_file, "GPU centroids")
    cpu_assignments = load_assignments(cpu_assignments_file, "CPU assignments")
    gpu_assignments = load_assignments(gpu_assignments_file, "GPU assignments")

    if (data_points is None or cpu_centroids is None or gpu_centroids is None or
        cpu_assignments is None or gpu_assignments is None):
        print("Skipping plot generation due to missing files.")
        return

    if data_points.shape[1] != 2:
        print("Error: Plotting function only supports 2-dimensional data. Skipping plot generation.")
        return
    
    if cpu_centroids.shape[1] != 2 or gpu_centroids.shape[1] != 2:
        print("Error: Centroids must be 2-dimensional for plotting. Skipping plot generation.")
        return

    # Assuming the number of clusters (K) is the same for both CPU and GPU
    # and assignments should match the number of points
    num_clusters_cpu = cpu_centroids.shape[0]
    num_clusters_gpu = gpu_centroids.shape[0]
    
    # Take the max cluster index to determine the range for coloring
    # This accounts for cases where K might be inferred differently
    max_cluster_id = max(np.max(cpu_assignments) if cpu_assignments.size > 0 else -1,
                         np.max(gpu_assignments) if gpu_assignments.size > 0 else -1)
    
    num_colors = max_cluster_id + 1 # Number of unique cluster IDs for palette

    try:
        import seaborn as sns
        palette = sns.color_palette("tab10", num_colors) # Ensure palette is large enough
        print("Using Seaborn color palette.")
    except ImportError:
        palette = plt.cm.get_cmap('tab10', num_colors)
        print("Seaborn not installed. Using default Matplotlib palette.")

    print("Creating plot figure...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('K-Means Comparison: CPU vs. GPU Results', fontsize=16)

    # Plot for CPU results
    print("Plotting results for CPU...")
    # Iterate through unique assignments to plot each cluster
    for i in range(num_colors): # Iterate through possible cluster IDs
        cluster_points = data_points[cpu_assignments == i]
        if cluster_points.size > 0:
            axes[0].scatter(cluster_points[:, 0], cluster_points[:, 1],
                            color=palette[i], label=f'Cluster {i} (CPU)', s=10, alpha=0.6)
    
    # Plot CPU centroids
    for i in range(num_clusters_cpu):
        axes[0].scatter(cpu_centroids[i, 0], cpu_centroids[i, 1],
                        marker='X', s=200, color=palette[i % num_colors], edgecolor='black', linewidth=1.5, zorder=5) # Use modulo for safety
    axes[0].set_title(f'K-Means (CPU) - {num_clusters_cpu} Clusters')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Plot for GPU results
    print("Plotting results for GPU...")
    for i in range(num_colors):
        cluster_points = data_points[gpu_assignments == i]
        if cluster_points.size > 0:
            axes[1].scatter(cluster_points[:, 0], cluster_points[:, 1],
                            color=palette[i], label=f'Cluster {i} (GPU)', s=10, alpha=0.6)
    
    # Plot GPU centroids
    for i in range(num_clusters_gpu):
        axes[1].scatter(gpu_centroids[i, 0], gpu_centroids[i, 1],
                        marker='X', s=200, color=palette[i % num_colors], edgecolor='black', linewidth=1.5, zorder=5)
    axes[1].set_title(f'K-Means (GPU) - {num_clusters_gpu} Clusters')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    print(f"Saving plot to file: {output_filename}...")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as: {output_filename}")
    plt.close(fig) # Close the figure to free up memory

    end_total_plot_time = time.time()
    print(f"--- Finished plot generation in {end_total_plot_time - start_total_plot_time:.2f} seconds ---")


if __name__ == "__main__":
    data_file = "data.txt"
    cpu_centroids_file = "centroids_cpu.txt"
    gpu_centroids_file = "centroids_gpu.txt"
    cpu_assignments_file = "assignments_cpu.txt" # Nowy plik z przypisaniami CPU
    gpu_assignments_file = "assignments_gpu.txt" # Nowy plik z przypisaniami GPU

    print("Checking for required files...")
    data_exists = os.path.exists(data_file)
    cpu_centroids_exists = os.path.exists(cpu_centroids_file)
    gpu_centroids_exists = os.path.exists(gpu_centroids_file)
    cpu_assignments_exists = os.path.exists(cpu_assignments_file)
    gpu_assignments_exists = os.path.exists(gpu_assignments_file)

    if not data_exists:
        print(f"Error: Data file '{data_file}' not found. Please ensure it has been generated.")
    if not cpu_centroids_exists:
        print(f"Error: File '{cpu_centroids_file}' not found. Please ensure you have run the CPU program.")
    if not gpu_centroids_exists:
        print(f"Error: File '{gpu_centroids_file}' not found. Please ensure you have run the GPU program.")
    if not cpu_assignments_exists:
        print(f"Error: File '{cpu_assignments_file}' not found. Please ensure the CPU program saves assignments.")
    if not gpu_assignments_exists:
        print(f"Error: File '{gpu_assignments_file}' not found. Please ensure the GPU program saves assignments.")
    
    # Only attempt to plot if all necessary files exist and data is 2D
    if (data_exists and cpu_centroids_exists and gpu_centroids_exists and
        cpu_assignments_exists and gpu_assignments_exists):
        
        # Load data points to check dimensionality before plotting
        temp_data = load_data_points(data_file)
        if temp_data is not None and temp_data.shape[1] == 2:
            plot_kmeans_results(data_file, cpu_centroids_file, gpu_centroids_file,
                                cpu_assignments_file, gpu_assignments_file,
                                'kmeans_comparison_plot.png')
        else:
            print(f"Skipping plot generation: Data in '{data_file}' is not 2-dimensional or the file is empty.")
    else:
        print("Skipping plot generation. Missing data, centroid, or assignment files.")