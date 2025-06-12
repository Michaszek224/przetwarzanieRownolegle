// Funkcje atomowe dla float (niezmienione)
inline void atomic_add_float_global(__global float *addr, float val) {
    union { unsigned int u; float f; } old, new;
    do {
        old.f = *addr;
        new.f = old.f + val;
    } while (atomic_cmpxchg((volatile __global unsigned int *)addr, old.u, new.u) != old.u);
}

inline void atomic_add_float_local(__local float *addr, float val) {
    union { unsigned int u; float f; } old, new;
    do {
        old.f = *addr;
        new.f = old.f + val;
    } while (atomic_cmpxchg((volatile __local unsigned int *)addr, old.u, new.u) != old.u);
}

// Kernele
__kernel void assign_to_clusters_kernel(
    __global const float* data_points,
    __global float* centroids,
    __global int* assignments,
    int num_points,
    int dimensions,
    int K,
    __local float* local_centroids)
{
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int local_size = get_local_size(0);

    // Załaduj centroidy do pamięci lokalnej
    for (int i = lid; i < K * dimensions; i += local_size) {
        local_centroids[i] = centroids[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid >= num_points) return;

    float min_dist_sq = FLT_MAX;
    int assigned_cluster_id = -1;

    for (int k = 0; k < K; ++k) {
        float current_dist_sq = 0.0f;
        for (int d = 0; d < dimensions; ++d) {
            float diff = data_points[gid * dimensions + d] - local_centroids[k * dimensions + d];
            current_dist_sq += diff * diff;
        }

        if (current_dist_sq < min_dist_sq) {
            min_dist_sq = current_dist_sq;
            assigned_cluster_id = k;
        }
    }
    assignments[gid] = assigned_cluster_id;
}

__kernel void reset_buffers_kernel(
    __global float* centroid_sums,
    __global int* centroid_counts,
    int K,
    int dimensions)
{
    int gid = get_global_id(0);
    if (gid < K * dimensions) centroid_sums[gid] = 0.0f;
    if (gid < K) centroid_counts[gid] = 0;
}

__kernel void accumulate_sums_kernel(
    __global const float* data_points,
    __global const int* assignments,
    __global float* centroid_sums,
    __global int* centroid_counts,
    int num_points,
    int dimensions,
    int K,
    __local float* local_sums,
    __local int* local_counts)
{
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int local_size = get_local_size(0);
    int num_clusters = K;

    // Resetuj lokalne bufory
    for (int i = lid; i < num_clusters * dimensions; i += local_size) {
        local_sums[i] = 0.0f;
    }
    for (int i = lid; i < num_clusters; i += local_size) {
        local_counts[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Akumuluj w pamięci lokalnej
    if (gid < num_points) {
        int cluster_id = assignments[gid];
        atomic_inc(local_counts + cluster_id);
        for (int d = 0; d < dimensions; ++d) {
            atomic_add_float_local(local_sums + cluster_id * dimensions + d, data_points[gid * dimensions + d]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Scal z buforami globalnymi
    for (int i = lid; i < num_clusters * dimensions; i += local_size) {
        atomic_add_float_global(&centroid_sums[i], local_sums[i]);
    }
    for (int i = lid; i < num_clusters; i += local_size) {
        atomic_add(&centroid_counts[i], local_counts[i]);
    }
}

__kernel void finalize_centroids_kernel(
    __global float* centroids,
    __global const float* centroid_sums,
    __global const int* centroid_counts,
    int dimensions,
    int K)
{
    int k_id = get_global_id(0);
    if (k_id >= K) return;

    if (centroid_counts[k_id] > 0) {
        for (int d = 0; d < dimensions; ++d) {
            centroids[k_id * dimensions + d] = centroid_sums[k_id * dimensions + d] / (float)centroid_counts[k_id];
        }
    }
}