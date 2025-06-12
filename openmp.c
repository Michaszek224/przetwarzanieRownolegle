#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <omp.h>

float* load_data_from_file_flat(const char* filename, int* num_points, int* dimensions) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Nie można otworzyć pliku danych: %s\n", filename);
        exit(1);
    }
    if (fscanf(file, "%d %d", num_points, dimensions) != 2) {
        fprintf(stderr, "Błąd odczytu liczby punktów i wymiarów z pliku: %s\n", filename);
        fclose(file);
        exit(1);
    }
    float* data_points_flat = (float*)malloc(*num_points * *dimensions * sizeof(float));
    if (!data_points_flat) {
        fprintf(stderr, "Błąd alokacji pamięci dla punktów danych.\n");
        fclose(file);
        exit(1);
    }
    for (int i = 0; i < *num_points * *dimensions; ++i) {
        if (fscanf(file, "%f", &data_points_flat[i]) != 1) {
            fprintf(stderr, "Błąd odczytu danych z pliku.\n");
            free(data_points_flat);
            fclose(file);
            exit(1);
        }
    }
    fclose(file);
    return data_points_flat;
}

void save_assignments_to_file(const char* filename, const int* assignments, int num_points) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Nie można otworzyć pliku do zapisu przypisań: %s\n", filename);
        return;
    }
    for (int i = 0; i < num_points; ++i) {
        fprintf(file, "%d\n", assignments[i]);
    }
    fclose(file);
    printf("Przypisania zapisane do pliku: %s\n", filename);
}

void save_centroids_to_file(const char* filename, const float* centroids_flat, int K, int dimensions) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Nie można otworzyć pliku do zapisu centroidów: %s\n", filename);
        return;
    }
    for (int i = 0; i < K; ++i) {
        for (int d = 0; d < dimensions; ++d) {
            fprintf(file, "%.6f%c", centroids_flat[i * dimensions + d], (d == dimensions - 1) ? '\n' : ' ');
        }
    }
    fclose(file);
    printf("Centroidy zapisane do pliku: %s\n", filename);
}

void initialize_centroids(const float* data_points_flat, int num_points, float* centroids_flat, int K, int dimensions) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < K; ++i) {
        int random_point_idx = rand() % num_points;
        for (int d = 0; d < dimensions; ++d) {
            centroids_flat[i * dimensions + d] = data_points_flat[random_point_idx * dimensions + d];
        }
    }
}

void assign_to_clusters(
    const float* data_points,
    const float* centroids,
    int* assignments,
    int num_points,
    int dimensions,
    int K)
{
    #pragma omp parallel for
    for (int gid = 0; gid < num_points; ++gid) {
        float min_dist_sq = FLT_MAX;
        int assigned_cluster_id = -1;
        for (int k = 0; k < K; ++k) {
            float current_dist_sq = 0.0f;
            for (int d = 0; d < dimensions; ++d) {
                float diff = data_points[gid * dimensions + d] - centroids[k * dimensions + d];
                current_dist_sq += diff * diff;
            }
            if (current_dist_sq < min_dist_sq) {
                min_dist_sq = current_dist_sq;
                assigned_cluster_id = k;
            }
        }
        assignments[gid] = assigned_cluster_id;
    }
}

void accumulate_sums(
    const float* data_points,
    const int* assignments,
    float* centroid_sums,
    int* centroid_counts,
    int num_points,
    int dimensions,
    int K)
{
    memset(centroid_sums, 0, K * dimensions * sizeof(float));
    for (int i = 0; i < K; ++i) {
        centroid_counts[i] = 0;
    }
    #pragma omp parallel
    {
        float* private_sums = (float*) calloc(K * dimensions, sizeof(float));
        int* private_counts = (int*) calloc(K, sizeof(int));
        #pragma omp for nowait
        for (int gid = 0; gid < num_points; ++gid) {
            int cluster_id = assignments[gid];
            private_counts[cluster_id]++;
            for (int d = 0; d < dimensions; ++d) {
                private_sums[cluster_id * dimensions + d] += data_points[gid * dimensions + d];
            }
        }
        #pragma omp critical
        {
            for (int k = 0; k < K; ++k) {
                centroid_counts[k] += private_counts[k];
                for (int d = 0; d < dimensions; ++d) {
                    centroid_sums[k * dimensions + d] += private_sums[k * dimensions + d];
                }
            }
        }
        free(private_sums);
        free(private_counts);
    }
}

void finalize_centroids(
    float* centroids,
    const float* centroid_sums,
    const int* centroid_counts,
    int dimensions,
    int K)
{
    #pragma omp parallel for
    for (int k_id = 0; k_id < K; ++k_id) {
        if (centroid_counts[k_id] > 0) {
            for (int d = 0; d < dimensions; ++d) {
                centroids[k_id * dimensions + d] = centroid_sums[k_id * dimensions + d] / (float)centroid_counts[k_id];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Użycie: %s <plik_danych.txt> [liczba_klastrow] [max_iteracji] [liczba_watkow]\n", argv[0]);
        return 1;
    }
    const char* data_filename = argv[1];
    int num_points, dimensions;
    int K = (argc > 2) ? atoi(argv[2]) : 5;
    int max_iterations = (argc > 3) ? atoi(argv[3]) : 10000;
    int num_threads = (argc > 4) ? atoi(argv[4]) : omp_get_max_threads();
    omp_set_num_threads(num_threads);
    printf("Użycie %d wątków OpenMP.\n", num_threads);
    float* h_data_points = load_data_from_file_flat(data_filename, &num_points, &dimensions);
    float* h_centroids = (float*)malloc(K * dimensions * sizeof(float));
    int* h_assignments = (int*)malloc(num_points * sizeof(int));
    float* h_centroid_sums = (float*)malloc(K * dimensions * sizeof(float));
    int* h_centroid_counts = (int*)malloc(K * sizeof(int));
    initialize_centroids(h_data_points, num_points, h_centroids, K, dimensions);
    printf("Rozpoczynanie obliczeń K-Means z %d klastrami i %d iteracjami...\n", K, max_iterations);
    double start_time = omp_get_wtime();
    for (int iter = 0; iter < max_iterations; iter++) {
        assign_to_clusters(h_data_points, h_centroids, h_assignments, num_points, dimensions, K);
        accumulate_sums(h_data_points, h_assignments, h_centroid_sums, h_centroid_counts, num_points, dimensions, K);
        finalize_centroids(h_centroids, h_centroid_sums, h_centroid_counts, dimensions, K);
    }
    double total_time = omp_get_wtime() - start_time;
    printf("Czas wykonania: %.4f s\n", total_time);
    save_centroids_to_file("centroids_openmp.txt", h_centroids, K, dimensions);
    save_assignments_to_file("assignments_openmp.txt", h_assignments, num_points);
    free(h_data_points);
    free(h_centroids);
    free(h_assignments);
    free(h_centroid_sums);
    free(h_centroid_counts);
    return 0;
}
