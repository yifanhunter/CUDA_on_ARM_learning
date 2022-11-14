#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define BLOCK_SIZE  8

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

__global__ void gpu_matrix_mult_shared(int *d_a, int *d_b, int *d_result, int m, int n, int k) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE]; // 共享变量
 
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x; // 横纵坐标
    int tmp = 0;
    int idx;
 
    for (int sub = 0; sub <= n/BLOCK_SIZE; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x; // 共享变量的对应于原数据的一维下标
       // printf("%d %d \n",sub * BLOCK_SIZE + threadIdx.x,n);
        tile_a[threadIdx.y][threadIdx.x] = row<m && (sub * BLOCK_SIZE + threadIdx.x)<n? d_a[idx]:0; // 规避多余操作
        idx = (sub * BLOCK_SIZE + threadIdx.y) * k + col; //原来的代码是乘以n，需要改成K
       // printf("%d %d %d %d %d\n",col,k,sub * BLOCK_SIZE + threadIdx.y, n,idx);
        tile_b[threadIdx.y][threadIdx.x] = col<k && (sub * BLOCK_SIZE + threadIdx.y)<n? d_b[idx]:0;
        /*
        一个共享块的例子：
        针对于tile_a
        共享变量的下标                              原数据下标:
        (0,0) (0,1) ... (0,15)                    0 1 2 ... 15
        (1,0) ...       (1,15)     --------->     n n+1 ... n+15
        ...                                       ...
        (15,0) ...      (15,15)                      15n 15n+1 ... 15n+15
        */
        
        __syncthreads(); //同步上述操作 求和
        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < m && col < k)
    {
        d_result[row * k + col] = tmp;
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
    int m=1000;
    int n=1000;
    int k=1000;

    int *h_a, *h_b, *h_c, *h_cc, *h_cs;
    CHECK(cudaMallocHost((void **) &h_a, sizeof(int)*m*n));
    CHECK(cudaMallocHost((void **) &h_b, sizeof(int)*n*k));
    CHECK(cudaMallocHost((void **) &h_c, sizeof(int)*m*k));
    CHECK(cudaMallocHost((void **) &h_cc, sizeof(int)*m*k));
    CHECK(cudaMallocHost((void **) &h_cs, sizeof(int)*m*k));
    
    cudaEvent_t start, stop,stop_share;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventCreate(&stop_share));
    
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
    

    int *d_a, *d_b, *d_c, *d_c_share;
    CHECK(cudaMalloc((void **) &d_a, sizeof(int)*m*n));
    CHECK(cudaMalloc((void **) &d_b, sizeof(int)*n*k));
    CHECK(cudaMalloc((void **) &d_c, sizeof(int)*m*k));
    CHECK(cudaMalloc((void **) &d_c_share, sizeof(int)*m*k));

    CHECK(cudaEventRecord(start));
    // copy matrix A and B from host to device memory
    CHECK(cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice));

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    

    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m,n,k);    

    CHECK(cudaMemcpy(h_c, d_c, (sizeof(int)*m*k), cudaMemcpyDeviceToHost));
    //cudaThreadSynchronize();
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    gpu_matrix_mult_shared<<<dimGrid, dimBlock>>>(d_a, d_b, d_c_share, m,n,k) ;
    CHECK(cudaMemcpy(h_cs, d_c_share, (sizeof(int)*m*k), cudaMemcpyDeviceToHost));
    
    CHECK(cudaEventRecord(stop_share));
    CHECK(cudaEventSynchronize(stop_share));
    
    float elapsed_time, elapsed_time_share;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    CHECK(cudaEventElapsedTime(&elapsed_time_share, stop, stop_share));
    printf("Time_global = %g ms.\n", elapsed_time);
    printf("Time_share = %g ms.\n", elapsed_time_share);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));    

    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    int ok = 1;
    for (int i = 0; i < m; ++i)
    { 
        for (int j = 0; j < k; ++j)
        {
            if(fabs(h_cs[i*k + j] - h_cc[i*k + j])>(1.0e-10))
            {
                printf("hcs: %d hc: %d  ",h_cs[i*k + j], h_c[i*k + j]);
                ok = 0;
            }
        }
    }
    
     printf("%d \n", h_cs[0]);
    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }
    
    // free memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_b));
    CHECK(cudaFreeHost(h_c));
    return 0;
}