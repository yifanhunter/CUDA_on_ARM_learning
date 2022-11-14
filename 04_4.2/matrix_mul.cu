#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define BLOCK_SIZE 32
__managed__ int a[1000 * 1000];
__managed__ int b[1000 * 1000];
__managed__ int u_gpu[1000 * 1000];
__managed__ int u_cpu[1000 * 1000];

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

	 //声明Event
    cudaEvent_t start, stop_cpu, stop_gpu;
	
	 //创建Event
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventCreate(&stop_gpu));
	
	//开辟主机空间
//	  int *h_a, *h_b, *h_c, *h_cc;
//    CHECK(cudaMallocHost((void **) &h_a, sizeof(int)*m*n));
//    CHECK(cudaMallocHost((void **) &h_b, sizeof(int)*n*k));
//    CHECK(cudaMallocHost((void **) &h_c, sizeof(int)*m*k));
//    CHECK(cudaMallocHost((void **) &h_cc, sizeof(int)*m*k));

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
	
	
	//int *d_a, *d_b, *d_c;
    //CHECK(cudaMalloc((void **) &d_a, sizeof(int)*m*n));
    //CHECK(cudaMalloc((void **) &d_b, sizeof(int)*n*k));
    //CHECK(cudaMalloc((void **) &d_c, sizeof(int)*m*k));

    // copy matrix A and B from host to device memory
    //CHECK(cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice));

	
	//开始start Event
    CHECK(cudaEventRecord(start));
    //非阻塞模式
    cudaEventQuery(start);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_matrix_mult_shared << <dimGrid, dimBlock >> > (a, b, u_gpu, m, n, k);
	//gpu_matrix_mult_shared<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k); 
	
	//CHECK(cudaMemcpy(h_c, d_c, (sizeof(int)*m*k), cudaMemcpyDeviceToHost));
    //cudaThreadSynchronize();

    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));

	// CPU 计算
    cpu_matrix_mult(a, b, u_cpu, m, n, k);
	//cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
	
	//记录时间消耗
    float elapsed_time_cpu, elapsed_time_gpu;
    CHECK(cudaEventElapsedTime(&elapsed_time_gpu, start, stop_gpu));
    CHECK(cudaEventElapsedTime(&elapsed_time_cpu, stop_gpu, stop_cpu));
    printf("GPU Time = %g ms.\n", elapsed_time_gpu);
    printf("CPU Time = %g ms.\n", elapsed_time_cpu);

	//销毁Event
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop_cpu));
    CHECK(cudaEventDestroy(stop_gpu));

	//对比正确性
    int ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if (fabs(u_gpu[i * k + j] - u_cpu[i * k + j]) > (1.0e-10))
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
	
	// free memory
    //cudaFree(d_a);
    //cudaFree(d_b);
    //cudaFree(d_c);
    //cudaFreeHost(h_a);
    //cudaFreeHost(h_b);
    //cudaFreeHost(h_c);
    //cudaFreeHost(h_cc);
    return 0;
}