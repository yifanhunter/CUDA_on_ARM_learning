#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define BLOCK_SIZE 16
__managed__ int a[1000 * 1000];
__managed__ int b[1000 * 1000];
__managed__ int c_gpu[1000 * 1000];
__managed__ int c_cpu[1000 * 1000];

__global__ void gpu_matrix_mult(int* a, int* b, int* c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}
__global__ void gpu_matrix_mult_shared(int* d_a, int* d_b, int* d_result, int M, int N, int K)
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub <= N/BLOCK_SIZE; ++sub)
    {
        int r = row;
        int c = sub * BLOCK_SIZE + threadIdx.x;
        idx = r * N + c;

        if (r >= M || c >= N)
        {
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        r = sub * BLOCK_SIZE + threadIdx.y;
        c = col;
        idx = r * K + c;
        if (c >= K || r >= N)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < K)
    {
        d_result[row * K + col] = tmp;
    }
}
void cpu_matrix_mult(int* a, int* b, int* h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += a[i * n + h] * b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const* argv[])
{
    int m = 1000;
    int n = 1000;
    int k = 1000;

    cudaEvent_t start, stop_cpu, stop_gpu;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventCreate(&stop_gpu));


    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = 0*rand() % 1024+1;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            b[i * k + j] = 0 * rand() % 1024 +1;
        }
    }

    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


    gpu_matrix_mult_shared << <dimGrid, dimBlock >> > (a, b, c_gpu, m, n, k);

    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));

    cpu_matrix_mult(a, b, c_cpu, m, n, k);
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    float elapsed_time_cpu, elapsed_time_gpu;
    CHECK(cudaEventElapsedTime(&elapsed_time_gpu, start, stop_gpu));
    CHECK(cudaEventElapsedTime(&elapsed_time_cpu, stop_gpu, stop_cpu));
    printf("GPU Time = %g ms.\n", elapsed_time_gpu);
    printf("CPU Time = %g ms.\n", elapsed_time_cpu);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop_cpu));
    CHECK(cudaEventDestroy(stop_gpu));

    

    int ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            //printf("GPU: % d; CPU: %d; ", h_c[i * k + j], h_cc[i * k + j]);
            if (fabs(c_gpu[i * k + j] - c_cpu[i * k + j]) > (1.0e-10))
            {

                ok = 0;
            }
            //printf("\n");
        }
    }

    if (ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    return 0;
}