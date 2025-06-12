#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <CL/cl.h>

void check_cl_error(cl_int err, const char* name) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: %s (%d)\n", name, err);
        exit(1);
    }
}

char* load_kernel_source(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Nie można otworzyć pliku kernela: %s\n", filename);
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* source = (char*)malloc(length + 1);
    if (!source) {
        fprintf(stderr, "Błąd alokacji pamięci dla źródła kernela.\n");
        fclose(file);
        exit(1);
    }
    size_t bytes_read = fread(source, 1, length, file);
    if (bytes_read != length) {
        fprintf(stderr, "Błąd odczytu całego pliku kernela: %s. Odczytano %zu z %zu bajtów.\n", filename, bytes_read, length);
        free(source);
        fclose(file);
        exit(1);
    }
    source[length] = '\0';
    fclose(file);
    return source;
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Użycie: %s <plik_danych.txt> [liczba_klastrow] [max_iteracji]\n", argv[0]);
        return 1;
    }
    const char* data_filename = argv[1];
    int num_points, dimensions;
    int K = (argc > 2) ? atoi(argv[2]) : 5;
    int max_iterations = (argc > 3) ? atoi(argv[3]) : 10000;
    float* h_data_points = load_data_from_file_flat(data_filename, &num_points, &dimensions);
    float* h_centroids = (float*)malloc(K * dimensions * sizeof(float));
    int* h_assignments = (int*)malloc(num_points * sizeof(int));
    initialize_centroids(h_data_points, num_points, h_centroids, K, dimensions);
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel assign_kernel, reset_kernel, accumulate_kernel, finalize_kernel;
    cl_int err;
    err = clGetPlatformIDs(1, &platform, NULL);
    check_cl_error(err, "clGetPlatformIDs");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "GPU nie znalezione, używam CPU\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        check_cl_error(err, "clGetDeviceIDs");
    }
    char device_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Urządzenie: %s\n", device_name);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_cl_error(err, "clCreateContext");
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    check_cl_error(err, "clCreateCommandQueue");
    char* kernel_source = load_kernel_source("gpu.cl");
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &err);
    check_cl_error(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Błąd kompilacji:\n%s\n", log);
        free(log);
        exit(1);
    }
    free(kernel_source);
    assign_kernel = clCreateKernel(program, "assign_to_clusters_kernel", &err);
    check_cl_error(err, "clCreateKernel assign");
    reset_kernel = clCreateKernel(program, "reset_buffers_kernel", &err);
    check_cl_error(err, "clCreateKernel reset");
    accumulate_kernel = clCreateKernel(program, "accumulate_sums_kernel", &err);
    check_cl_error(err, "clCreateKernel accumulate");
    finalize_kernel = clCreateKernel(program, "finalize_centroids_kernel", &err);
    check_cl_error(err, "clCreateKernel finalize");
    size_t data_size = num_points * dimensions * sizeof(float);
    size_t centroid_size = K * dimensions * sizeof(float);
    size_t assignment_size = num_points * sizeof(int);
    size_t sum_size = K * dimensions * sizeof(float);
    size_t count_size = K * sizeof(int);
    cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data_size, h_data_points, &err);
    cl_mem d_centroids = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, centroid_size, h_centroids, &err);
    cl_mem d_assignments = clCreateBuffer(context, CL_MEM_READ_WRITE, assignment_size, NULL, &err);
    cl_mem d_sums = clCreateBuffer(context, CL_MEM_READ_WRITE, sum_size, NULL, &err);
    cl_mem d_counts = clCreateBuffer(context, CL_MEM_READ_WRITE, count_size, NULL, &err);
    size_t local_assign[1] = {256};
    size_t global_assign[1] = {((num_points + 255) / 256) * 256};
    size_t local_accumulate[1] = {256};
    size_t global_accumulate[1] = {((num_points + 255) / 256) * 256};
    size_t local_finalize[1] = {K < 256 ? K : 256};
    size_t global_finalize[1] = {K};
    size_t local_centroid_mem = K * dimensions * sizeof(float);
    size_t local_sum_mem = K * dimensions * sizeof(float);
    size_t local_count_mem = K * sizeof(int);
    clock_t start = clock();
    double kernel_time = 0;
    for (int iter = 0; iter < max_iterations; iter++) {
        clSetKernelArg(assign_kernel, 0, sizeof(cl_mem), &d_data);
        clSetKernelArg(assign_kernel, 1, sizeof(cl_mem), &d_centroids);
        clSetKernelArg(assign_kernel, 2, sizeof(cl_mem), &d_assignments);
        clSetKernelArg(assign_kernel, 3, sizeof(int), &num_points);
        clSetKernelArg(assign_kernel, 4, sizeof(int), &dimensions);
        clSetKernelArg(assign_kernel, 5, sizeof(int), &K);
        clSetKernelArg(assign_kernel, 6, local_centroid_mem, NULL); 
        cl_event assign_event;
        clEnqueueNDRangeKernel(queue, assign_kernel, 1, NULL, global_assign, local_assign, 0, NULL, &assign_event);
        clWaitForEvents(1, &assign_event);
        clSetKernelArg(reset_kernel, 0, sizeof(cl_mem), &d_sums);
        clSetKernelArg(reset_kernel, 1, sizeof(cl_mem), &d_counts);
        clSetKernelArg(reset_kernel, 2, sizeof(int), &K);
        clSetKernelArg(reset_kernel, 3, sizeof(int), &dimensions);
        size_t global_reset_size = ((K * dimensions + local_assign[0] - 1) / local_assign[0]) * local_assign[0];
        if (K > global_reset_size) {
             global_reset_size = ((K + local_assign[0] - 1) / local_assign[0]) * local_assign[0];
        }
        clEnqueueNDRangeKernel(queue, reset_kernel, 1, NULL, &global_reset_size, local_assign, 0, NULL, NULL);
        clSetKernelArg(accumulate_kernel, 0, sizeof(cl_mem), &d_data);
        clSetKernelArg(accumulate_kernel, 1, sizeof(cl_mem), &d_assignments);
        clSetKernelArg(accumulate_kernel, 2, sizeof(cl_mem), &d_sums);
        clSetKernelArg(accumulate_kernel, 3, sizeof(cl_mem), &d_counts);
        clSetKernelArg(accumulate_kernel, 4, sizeof(int), &num_points);
        clSetKernelArg(accumulate_kernel, 5, sizeof(int), &dimensions);
        clSetKernelArg(accumulate_kernel, 6, sizeof(int), &K);
        clSetKernelArg(accumulate_kernel, 7, local_sum_mem, NULL); 
        clSetKernelArg(accumulate_kernel, 8, local_count_mem, NULL);
        clEnqueueNDRangeKernel(queue, accumulate_kernel, 1, NULL, global_accumulate, local_accumulate, 0, NULL, NULL);
        clSetKernelArg(finalize_kernel, 0, sizeof(cl_mem), &d_centroids);
        clSetKernelArg(finalize_kernel, 1, sizeof(cl_mem), &d_sums);
        clSetKernelArg(finalize_kernel, 2, sizeof(cl_mem), &d_counts);
        clSetKernelArg(finalize_kernel, 3, sizeof(int), &dimensions);
        clSetKernelArg(finalize_kernel, 4, sizeof(int), &K);
        clEnqueueNDRangeKernel(queue, finalize_kernel, 1, NULL, global_finalize, local_finalize, 0, NULL, NULL);
    }
    clFinish(queue);
    double total_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("Czas wykonania: %.4f s\n", total_time);
    clEnqueueReadBuffer(queue, d_centroids, CL_TRUE, 0, centroid_size, h_centroids, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_assignments, CL_TRUE, 0, assignment_size, h_assignments, 0, NULL, NULL);
    save_centroids_to_file("centroids_gpu.txt", h_centroids, K, dimensions);
    save_assignments_to_file("assignments_gpu.txt", h_assignments, num_points);
    clReleaseMemObject(d_data);
    clReleaseMemObject(d_centroids);
    clReleaseMemObject(d_assignments);
    clReleaseMemObject(d_sums);
    clReleaseMemObject(d_counts);
    clReleaseKernel(assign_kernel);
    clReleaseKernel(reset_kernel);
    clReleaseKernel(accumulate_kernel);
    clReleaseKernel(finalize_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_data_points);
    free(h_centroids);
    free(h_assignments);
    return 0;
}
