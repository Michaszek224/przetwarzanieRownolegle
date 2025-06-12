#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>

typedef struct {
    float* coords;
    int dimensions;
} Point;

typedef struct {
    float* coords;
    int dimensions;
    int count;
} Centroid;

void save_assignments_to_file(const char* filename, const int* assignments, int num_points) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Błąd: Nie można otworzyć pliku do zapisu przypisań: %s\n", filename);
        return;
    }
    for (int i = 0; i < num_points; ++i) {
        fprintf(file, "%d\n", assignments[i]);
    }
    fclose(file);
    printf("Przypisania punktów zapisane do pliku: %s\n", filename);
}

void save_centroids_to_file(const char* filename, const float* centroids_flat, int K, int dimensions) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Błąd: Nie można otworzyć pliku do zapisu: %s\n", filename);
        return;
    }
    for (int i = 0; i < K; ++i) {
        for (int d = 0; d < dimensions; ++d) {
            fprintf(file, "%.6f%s", centroids_flat[i * dimensions + d], (d == dimensions - 1) ? "" : " ");
        }
        fprintf(file, "\n");
    }
    fclose(file);
    printf("Końcowe centroidy zapisane do pliku: %s\n", filename);
}

float euclidean_distance(const Point* p1, const float* p2_coords) {
    float dist = 0.0f;
    for (int i = 0; i < p1->dimensions; ++i) {
        dist += powf(p1->coords[i] - p2_coords[i], 2.0f);
    }
    return sqrtf(dist);
}

void initialize_centroids(Point* data_points, int num_points, Centroid* centroids, int K, int dimensions) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < K; ++i) {
        int random_index = rand() % num_points;
        centroids[i].coords = (float*)malloc(dimensions * sizeof(float));
        if (centroids[i].coords == NULL) {
            fprintf(stderr, "Błąd alokacji pamięci dla centroidu.\n");
            exit(EXIT_FAILURE);
        }
        centroids[i].dimensions = dimensions;
        centroids[i].count = 0;
        for (int d = 0; d < dimensions; ++d) {
            centroids[i].coords[d] = data_points[random_index].coords[d];
        }
    }
}

int kmeans_iteration(Point* data_points, int num_points, Centroid* centroids, int K, int* assignments) {
    int changed = 0;
    for (int i = 0; i < num_points; ++i) {
        float min_dist = FLT_MAX;
        int closest_centroid_idx = -1;
        for (int j = 0; j < K; ++j) {
            float dist = euclidean_distance(&data_points[i], centroids[j].coords);
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid_idx = j;
            }
        }
        if (assignments[i] != closest_centroid_idx) {
            assignments[i] = closest_centroid_idx;
            changed = 1;
        }
    }
    if (changed == 0) {
        return 0;
    }
    for (int j = 0; j < K; ++j) {
        centroids[j].count = 0;
        for (int d = 0; d < centroids[j].dimensions; ++d) {
            centroids[j].coords[d] = 0.0f;
        }
    }
    for (int i = 0; i < num_points; ++i) {
        int assigned_centroid_idx = assignments[i];
        centroids[assigned_centroid_idx].count++;
        for (int d = 0; d < centroids[assigned_centroid_idx].dimensions; ++d) {
            centroids[assigned_centroid_idx].coords[d] += data_points[i].coords[d];
        }
    }
    for (int j = 0; j < K; ++j) {
        if (centroids[j].count > 0) {
            for (int d = 0; d < centroids[j].dimensions; ++d) {
                centroids[j].coords[d] /= centroids[j].count;
            }
        }
    }
    return 1;
}

Point* load_data_from_file(const char* filename, int* num_points, int* dimensions) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Błąd: Nie można otworzyć pliku danych '%s'\n", filename);
        exit(EXIT_FAILURE);
    }
    if (fscanf(file, "%d %d", num_points, dimensions) != 2) {
        fprintf(stderr, "Błąd: Nie udało się odczytać liczby punktów i wymiarów z pierwszej linii pliku '%s'.\n", filename);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    Point* data_points = (Point*)malloc(*num_points * sizeof(Point));
    if (data_points == NULL) {
        fprintf(stderr, "Błąd alokacji pamięci dla punktów danych.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < *num_points; ++i) {
        data_points[i].dimensions = *dimensions;
        data_points[i].coords = (float*)malloc(*dimensions * sizeof(float));
        if (data_points[i].coords == NULL) {
            fprintf(stderr, "Błąd alokacji pamięci dla współrzędnych punktu.\n");
            for (int j = 0; j < i; ++j) {
                free(data_points[j].coords);
            }
            free(data_points);
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int d = 0; d < *dimensions; ++d) {
            if (fscanf(file, "%f", &data_points[i].coords[d]) != 1) {
                fprintf(stderr, "Błąd odczytu danych z pliku '%s' w punkcie %d, wymiar %d. Sprawdź format pliku.\n", filename, i, d);
                for (int j = 0; j <= i; ++j) {
                    free(data_points[j].coords);
                }
                free(data_points);
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
    printf("Wczytano %d punktów o %d wymiarach z pliku '%s'.\n", *num_points, *dimensions, filename);
    return data_points;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Użycie: %s <plik_danych.txt> [liczba_klastrow] [max_iteracji]\n", argv[0]);
        return 1;
    }
    const char* data_filename = argv[1];
    int num_points;
    int dimensions;
    int K = (argc > 2) ? atoi(argv[2]) : 5;
    int max_iterations = (argc > 3) ? atoi(argv[3]) : 10000;
    Point* data_points = load_data_from_file(data_filename, &num_points, &dimensions);
    Centroid* centroids = (Centroid*)malloc(K * sizeof(Centroid));
    if (centroids == NULL) {
        fprintf(stderr, "Błąd alokacji pamięci dla centroidów.\n");
        for (int i = 0; i < num_points; ++i) {
            free(data_points[i].coords);
        }
        free(data_points);
        return 1;
    }
    int* assignments = (int*)malloc(num_points * sizeof(int));
    if (assignments == NULL) {
        fprintf(stderr, "Błąd alokacji pamięci dla przypisań.\n");
        for (int i = 0; i < num_points; ++i) {
            free(data_points[i].coords);
        }
        free(data_points);
        for (int i = 0; i < K; ++i) {
            if (centroids[i].coords) free(centroids[i].coords);
        }
        free(centroids);
        return 1;
    }
    for (int i = 0; i < num_points; ++i) {
        assignments[i] = -1;
    }
    initialize_centroids(data_points, num_points, centroids, K, dimensions);
    printf("Centroidy początkowe zostały zainicjalizowane.\n");
    clock_t start_time, end_time;
    double cpu_time_used;
    printf("Rozpoczynanie obliczeń K-Means sekwencyjnie...\n");
    start_time = clock();
    int iteration = 0;
    int changed_assignments = 1;
    while (iteration < max_iterations && changed_assignments) {
        printf("Iteracja %d...\n", iteration + 1);
        changed_assignments = kmeans_iteration(data_points, num_points, centroids, K, assignments);
        iteration++;
    }
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Obliczenia K-Means zakończone po %d iteracjach.\n", iteration);
    printf("Czas wykonania na CPU: %f sekund\n", cpu_time_used);
    printf("\nKońcowe pozycje centroidów:\n");
    for (int i = 0; i < K; ++i) {
        printf("Centroid %d: (", i);
        for (int d = 0; d < dimensions; ++d) {
            printf("%.4f%s", centroids[i].coords[d], (d == dimensions - 1) ? "" : ", ");
        }
        printf(")\n");
    }
    float* h_centroids_flat = (float*)malloc(K * dimensions * sizeof(float));
    if (h_centroids_flat == NULL) {
        fprintf(stderr, "Błąd alokacji pamięci dla płaskich centroidów CPU.\n");
    }
    for (int i = 0; i < K; ++i) {
        for (int d = 0; d < dimensions; ++d) {
            h_centroids_flat[i * dimensions + d] = centroids[i].coords[d];
        }
    }
    save_centroids_to_file("centroids_cpu.txt", h_centroids_flat, K, dimensions);
    free(h_centroids_flat);
    if (h_centroids_flat == NULL) {
        fprintf(stderr, "Błąd alokacji pamięci dla płaskich centroidów CPU.\n");
    }
    for (int i = 0; i < K; ++i) {
        for (int d = 0; d < dimensions; ++d) {
            h_centroids_flat[i * dimensions + d] = centroids[i].coords[d];
        }
    }
    save_centroids_to_file("centroids_cpu.txt", h_centroids_flat, K, dimensions);
    free(h_centroids_flat);
    save_assignments_to_file("assignments_cpu.txt", assignments, num_points);
    for (int i = 0; i < num_points; ++i) {
        free(data_points[i].coords);
    }
    free(data_points);
    for (int i = 0; i < K; ++i) {
        free(centroids[i].coords);
    }
    free(centroids);
    free(assignments);
    return 0;
}
