import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

def load_data_points(filename):
    print(f"Loading data points from file: {filename}...")
    start_time = time.time()
    try:
        with open(filename, 'r') as f:
            header = f.readline().strip().split()
            if len(header) != 2:
                print(f"Error: Invalid header in '{filename}'. Expected 'num_points dimensions'.")
                return None
            
            num_points_expected = int(header[0])
            dimensions_expected = int(header[1])
            data = np.loadtxt(f, dtype=np.float32)
            if data.shape[0] != num_points_expected or data.shape[1] != dimensions_expected:
                print(f"Warning: Data shape mismatch in '{filename}'. Expected ({num_points_expected}, {dimensions_expected}), got {data.shape}.")

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
    print(f"Loading {type_name} from file: {filename}...")
    start_time = time.time()
    try:
        data = np.loadtxt(filename, dtype=np.float32)
        end_time = time.time()
        print(f"Loaded {data.shape[0]} {type_name} in {end_time - start_time:.2f} seconds.")
        if data.ndim == 1: # Handle case where there's only one centroid
            data = data.reshape(1, -1)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading file '{filename}': {e}")
        return None

def load_assignments(filename, type_name="assignments"):
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

def plot_kmeans_results_three_ways(data_points_file, 
                                   sequential_centroids_file, sequential_assignments_file,
                                   openmp_centroids_file, openmp_assignments_file,
                                   gpu_centroids_file, gpu_assignments_file,
                                   output_filename='kmeans_three_way_comparison_plot.png'):
    print("\n--- Starting plot generation for Sequential, OpenMP, and GPU ---")
    start_total_plot_time = time.time()

    data_points = load_data_points(data_points_file)
    
    sequential_centroids = load_centroids(sequential_centroids_file, "Sequential centroids")
    sequential_assignments = load_assignments(sequential_assignments_file, "Sequential assignments")
    
    openmp_centroids = load_centroids(openmp_centroids_file, "OpenMP centroids")
    openmp_assignments = load_assignments(openmp_assignments_file, "OpenMP assignments")
    
    gpu_centroids = load_centroids(gpu_centroids_file, "GPU (OpenCL) centroids")
    gpu_assignments = load_assignments(gpu_assignments_file, "GPU (OpenCL) assignments")

    # Check if all necessary files were loaded successfully
    if (data_points is None or 
        sequential_centroids is None or sequential_assignments is None or
        openmp_centroids is None or openmp_assignments is None or
        gpu_centroids is None or gpu_assignments is None):
        print("Skipping plot generation due to missing or incorrectly loaded files.")
        return

    # Check for 2D data for plotting
    if data_points.shape[1] != 2:
        print(f"Error: Plotting function only supports 2-dimensional data. Data has {data_points.shape[1]} dimensions. Skipping plot generation.")
        return
    
    if sequential_centroids.shape[1] != 2 or openmp_centroids.shape[1] != 2 or gpu_centroids.shape[1] != 2:
        print("Error: Centroids must be 2-dimensional for plotting. Skipping plot generation.")
        return

    # Determine number of clusters and colors for palette
    num_clusters_seq = sequential_centroids.shape[0]
    num_clusters_openmp = openmp_centroids.shape[0]
    num_clusters_gpu = gpu_centroids.shape[0]

    # Ensure all assignments and centroids have valid cluster IDs to determine max_cluster_id
    all_assignments = []
    if sequential_assignments.size > 0: all_assignments.append(np.max(sequential_assignments))
    if openmp_assignments.size > 0: all_assignments.append(np.max(openmp_assignments))
    if gpu_assignments.size > 0: all_assignments.append(np.max(gpu_assignments))

    max_cluster_id = max(all_assignments) if all_assignments else -1
    num_colors = max_cluster_id + 1
    
    if num_colors <= 0:
        print("Warning: No valid cluster assignments found. Cannot generate plot with colors.")
        num_colors = 1
    
    try:
        # Use Seaborn if available, otherwise fallback to Matplotlib's default
        import seaborn as sns
        palette = sns.color_palette("tab10", num_colors)
        print(f"Using Seaborn color palette with {num_colors} colors.")
    except ImportError:
        palette = plt.cm.get_cmap('tab10', num_colors)
        print(f"Seaborn not installed. Using default Matplotlib 'tab10' palette with {num_colors} colors.")

    print("Creating plot figure with three subplots...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7)) # Changed to 3 subplots
    fig.suptitle('K-Means Comparison: Sequential vs. OpenMP vs. GPU (OpenCL)', fontsize=16)

    # --- Plot for Sequential (CPU) ---
    print("Plotting results for Sequential (CPU)...")
    for i in range(num_colors):
        cluster_points = data_points[sequential_assignments == i]
        if cluster_points.size > 0:
            axes[0].scatter(cluster_points[:, 0], cluster_points[:, 1],
                            color=palette[i], label=f'Cluster {i}', s=10, alpha=0.6)
    for i in range(num_clusters_seq):
        color_idx = i % num_colors 
        axes[0].scatter(sequential_centroids[i, 0], sequential_centroids[i, 1],
                        marker='X', s=200, color=palette[color_idx], edgecolor='black', linewidth=1.5, zorder=5) 
    axes[0].set_title(f'K-Means (Sequential) - {num_clusters_seq} Clusters')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # --- Plot for OpenMP ---
    print("Plotting results for OpenMP (CPU)...")
    for i in range(num_colors):
        cluster_points = data_points[openmp_assignments == i]
        if cluster_points.size > 0:
            axes[1].scatter(cluster_points[:, 0], cluster_points[:, 1],
                            color=palette[i], label=f'Cluster {i}', s=10, alpha=0.6)
    for i in range(num_clusters_openmp):
        color_idx = i % num_colors
        axes[1].scatter(openmp_centroids[i, 0], openmp_centroids[i, 1],
                        marker='X', s=200, color=palette[color_idx], edgecolor='black', linewidth=1.5, zorder=5)
    axes[1].set_title(f'K-Means (OpenMP) - {num_clusters_openmp} Clusters')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # --- Plot for GPU (OpenCL) ---
    print("Plotting results for GPU (OpenCL)...")
    for i in range(num_colors):
        cluster_points = data_points[gpu_assignments == i]
        if cluster_points.size > 0:
            axes[2].scatter(cluster_points[:, 0], cluster_points[:, 1],
                            color=palette[i], label=f'Cluster {i}', s=10, alpha=0.6)
    for i in range(num_clusters_gpu):
        color_idx = i % num_colors
        axes[2].scatter(gpu_centroids[i, 0], gpu_centroids[i, 1],
                        marker='X', s=200, color=palette[color_idx], edgecolor='black', linewidth=1.5, zorder=5)
    axes[2].set_title(f'K-Means (GPU - OpenCL) - {num_clusters_gpu} Clusters')
    axes[2].set_xlabel('Dimension 1')
    axes[2].set_ylabel('Dimension 2')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    print(f"Saving plot to file: {output_filename}...")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as: {output_filename}")
    plt.close(fig)

    end_total_plot_time = time.time()
    print(f"--- Finished plot generation in {end_total_plot_time - start_total_plot_time:.2f} seconds ---")

if __name__ == "__main__":
    # Common data file
    data_file = "data.txt"

    # Files for Sequential implementation
    sequential_centroids_file = "centroids_cpu.txt"
    sequential_assignments_file = "assignments_cpu.txt"

    # Files for OpenMP implementation
    openmp_centroids_file = "centroids_openmp.txt" 
    openmp_assignments_file = "assignments_openmp.txt"
    
    # Files for GPU (OpenCL) implementation
    gpu_centroids_file = "centroids_gpu.txt" 
    gpu_assignments_file = "assignments_gpu.txt" 

    print("Checking for required files...")
    data_exists = os.path.exists(data_file)
    
    sequential_centroids_exists = os.path.exists(sequential_centroids_file)
    sequential_assignments_exists = os.path.exists(sequential_assignments_file)
    
    openmp_centroids_exists = os.path.exists(openmp_centroids_file)
    openmp_assignments_exists = os.path.exists(openmp_assignments_file)
    
    gpu_centroids_exists = os.path.exists(gpu_centroids_file)
    gpu_assignments_exists = os.path.exists(gpu_assignments_file)

    # --- File Existence Checks ---
    if not data_exists:
        print(f"Error: Data file '{data_file}' not found. Please ensure it has been generated.")
    if not sequential_centroids_exists:
        print(f"Error: File '{sequential_centroids_file}' not found. Please ensure you have run the Sequential program.")
    if not sequential_assignments_exists:
        print(f"Error: File '{sequential_assignments_file}' not found. Please ensure the Sequential program saves assignments.")
    if not openmp_centroids_exists:
        print(f"Error: File '{openmp_centroids_file}' not found. Please ensure you have run the OpenMP program.")
    if not openmp_assignments_exists:
        print(f"Error: File '{openmp_assignments_file}' not found. Please ensure the OpenMP program saves assignments.")
    if not gpu_centroids_exists:
        print(f"Error: File '{gpu_centroids_file}' not found. Please ensure you have run the GPU (OpenCL) program.")
    if not gpu_assignments_exists:
        print(f"Error: File '{gpu_assignments_file}' not found. Please ensure the GPU (OpenCL) program saves assignments.")
    
    # Proceed with plotting only if all required files exist
    if (data_exists and 
        sequential_centroids_exists and sequential_assignments_exists and
        openmp_centroids_exists and openmp_assignments_exists and
        gpu_centroids_exists and gpu_assignments_exists):
        
        temp_data = load_data_points(data_file)
        if temp_data is not None and temp_data.shape[1] == 2:
            plot_kmeans_results_three_ways(
                data_file, 
                sequential_centroids_file, sequential_assignments_file,
                openmp_centroids_file, openmp_assignments_file,
                gpu_centroids_file, gpu_assignments_file,
                'kmeans_three_way_comparison_plot.png' # Changed output filename
            )
        else:
            print(f"Skipping plot generation: Data in '{data_file}' is not 2-dimensional or the file is empty/corrupt.")
    else:
        print("Skipping plot generation. Missing data, centroid, or assignment files.")