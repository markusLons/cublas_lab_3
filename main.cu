
#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define at(arr, x, y) (arr[(x) * size + (y)])
#define maximum_t 32

double epsilon = 1E-6;
int maximum_iteration = 1E6;
int size = 128;
bool show_matrix = false;
int update_matrix = 100;

__global__ void computeOneStepOnGPU(double *A, double *B, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // чтобы не изменять границы
    if (j == 0 || i == 0 || i >= size-1 || j >= size-1)
        return;

    at(B, i, j) = 0.25 * (at(A, i-1, j) + at(A, i, j-1) + at(A, i+1, j) + at(A, i, j+1));
}

__global__ void gpuMatrixSubtraction(double *A, double *B, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if( i >= (size-1))
        return;
    if (j >= (size-1))
        return;
    at(B, i, j) = at(A, i, j) - at(B, i, j);
}

__global__ void gpuInitializeMatrixA(double *A, int size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= size-1)
        return;

    int step;
    step = 10.0 / (size - 1) * i;

    at(A, 0, i) = 10.0 + step;
    at(A, i, 0) = 10.0 + step;
    at(A, i, size-1) = 20.0 + step;
    at(A, size-1, i) = 20.0 + step;
}

int main(int argc, char **argv) {
    cudaSetDevice(3);
    for(int arg = 0; arg < argc; arg += 1) {
        if(0 == strcmp(argv[arg], "-error")) {
            epsilon = atof(argv[arg + 1]);
            arg += 1;
        }
        else if(0 == strcmp(argv[arg], "-iter")) {
            maximum_iteration = atoi(argv[arg + 1]);
            arg += 1;
        }
        else if(0 == strcmp(argv[arg], "-size")) {
            size = atoi(argv[arg+1]);
            arg += 1;
        }
        else if(0 == strcmp(argv[arg], "-update_matrix")) {
            update_matrix = atoi(argv[arg + 1]);
            arg += 1;
        }
        else if(0 == strcmp(argv[arg], "-show_matrix")) {
            show_matrix = true;
            arg += 1;
        }
    }

    int size_matrix = size * size;

    cudaError_t crush;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t     graph;
    cudaGraphExec_t g_instance;

    double *A_GPU;
    double *cudaError;
    double *tmp = NULL;
    double *A_GPU_NEW;

    // количество блоков в гриде и нитей в блоке соответственно
    dim3 blocks_in_grid   = dim3(size / maximum_t, size / maximum_t);
    dim3 threads_in_block = dim3(maximum_t, maximum_t);

    size_t temp_size = 0;

    cudaMalloc(&cudaError, sizeof(double) * 1);
    cudaMalloc(&A_GPU, sizeof(double) * size_matrix);
    cudaMalloc(&A_GPU_NEW, sizeof(double) * size_matrix);

    dim3 thread = size < 1024 ? size : 1024;
    dim3 block = size / (size < 1024 ? size : 1024);

    // заполнение матриц начальными значениями (значения границ)
    gpuInitializeMatrixA<<<block, thread>>>(A_GPU, size);
    gpuInitializeMatrixA<<<block, thread>>>(A_GPU_NEW, size);

    // вычисление tmp_size (размера tmp)
    cub::DeviceReduce::Max(tmp, temp_size, A_GPU_NEW, cudaError, size_matrix, stream);
    cudaMalloc(&tmp, temp_size);


    // ================================================================================
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // выполнение шагов алгоритма столько раз, сколько необходимо до обновления ошибки
    for(int i = 0; i < update_matrix; i += 2)
    {
        computeOneStepOnGPU<<<blocks_in_grid, threads_in_block, 0, stream>>>(A_GPU, A_GPU_NEW, size);
        computeOneStepOnGPU<<<blocks_in_grid, threads_in_block, 0, stream>>>(A_GPU_NEW, A_GPU, size);
    }

    gpuMatrixSubtraction<<<blocks_in_grid, threads_in_block, 0, stream>>>(A_GPU, A_GPU_NEW, size);

    // вычисление ошибки
    cub::DeviceReduce::Max(tmp, temp_size, A_GPU_NEW, cudaError, size_matrix, stream);

    gpuInitializeMatrixA<<<block, thread, 0, stream>>>(A_GPU_NEW, size);

    cudaStreamEndCapture(stream, &graph);

    cudaGraphInstantiate(&g_instance, graph, NULL, NULL, 0);

    // =================================================================================

    int iter = 0;
    double error = 1.0;

    while(error > epsilon && iter < maximum_iteration)
    {
        cudaGraphLaunch(g_instance, stream);
        cudaMemcpy(&error, cudaError, sizeof(double), cudaMemcpyDeviceToHost);
        iter += update_matrix;
    }

    std::cout << "iterations: " << iter  << std::endl;
    std::cout << "error: " << error << std::endl;

    // вывод матрицы
    if(show_matrix)
    {
        double* A = new double[size_matrix];
        cudaMemcpyAsync(&A, A_GPU, sizeof(double), cudaMemcpyDeviceToHost, stream);
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++)
                std::cout << at(A_GPU, i, j) << ' ';
            std::cout << std::endl;
        }
    }

    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaFree(A_GPU);
    cudaFree(A_GPU_NEW);
    cudaFree(tmp);

    return 0;
}
