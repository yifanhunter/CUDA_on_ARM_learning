#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult_shared(int *d_a, int *d_b, int *d_result, int n) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
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
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}


void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    int m=100;
    int n=100;
    int k=100;

    int *h_a, *h_b, *h_c, *h_cc;
    CHECK(cudaMallocHost((void **) &h_a, sizeof(int)*m*n));
    CHECK(cudaMallocHost((void **) &h_b, sizeof(int)*n*k));
    CHECK(cudaMallocHost((void **) &h_c, sizeof(int)*m*k));
    CHECK(cudaMallocHost((void **) &h_cc, sizeof(int)*m*k));
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));


    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }
    
    CHECK(cudaEventRecord(start));
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    gpu_matrix_mult_shared<<<dimGrid, dimBlock>>>(h_a, h_b, h_c, k);    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time = %g ms.\n", elapsed_time);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    
    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    int ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if(fabs(h_cc[i*k + j] - h_c[i*k + j])>(1.0e-10))
            {
                
                ok = 0;
            }
        }
    }

    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    // free memory
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_b));
    CHECK(cudaFreeHost(h_c));
    return 0;
}