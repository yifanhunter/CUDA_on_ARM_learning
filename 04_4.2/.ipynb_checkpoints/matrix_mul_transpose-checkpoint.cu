#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#define BLOCK_SIZE 32
using namespace std;


__global__ void cuda_transpose(int *matrix,int *tr_matrix,int m,int n) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ int smem_matrix[BLOCK_SIZE][BLOCK_SIZE];
	smem_matrix[threadIdx.y][threadIdx.x] = row < m&& col < n ? matrix[row*n+col] : 0;
	__syncthreads();
	if(blockIdx.x * blockDim.x + threadIdx.y < n && threadIdx.x + blockIdx.y * blockDim.x < m)
	tr_matrix[threadIdx.x+blockIdx.y*blockDim.x+m*(blockIdx.x*blockDim.x+threadIdx.y)] = smem_matrix[threadIdx.x][threadIdx.y];
	return;
}

__host__ void cpu_transpose(int *matrix,int *tr_matrix,int m,int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			tr_matrix[i * m + j] = matrix[j * n + i];
		}
	}
	return;
}

__host__ void init_matrix(int* matrix,int m,int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			matrix[i*n+j] = rand();
		}
	}
}

void print(int*, string,int,int);
bool check(int*, int*, int, int);

int main() {
	int m = 1111;
	int n = 11312;
	int *matrix;
	cudaMallocHost((void**)&matrix, sizeof(int) * m * n);
	init_matrix(matrix,m,n);
	//print(matrix, "init matrix", m, n);


	int* htr_matrix;
	cudaMallocHost((void**)&htr_matrix, sizeof(int) * m * n);
	cpu_transpose(matrix, htr_matrix, m, n);
	//print(htr_matrix, "CPU", n, m);
	//将CPU端执行的结果存放在htr_matrix中 

	int* d_matrix, *dtr_matrix;
	cudaMalloc((void**)&d_matrix, sizeof(int) * m * n);
	cudaMalloc((void**)&dtr_matrix, sizeof(int) * m * n);
	cudaMemcpy(d_matrix, matrix, sizeof(int) * m * n, cudaMemcpyHostToDevice);
	dim3 gird = { (unsigned int)(n - 1 + BLOCK_SIZE) / BLOCK_SIZE, (unsigned int)(m - 1 + BLOCK_SIZE) / BLOCK_SIZE,1 };
	dim3 block = { BLOCK_SIZE,BLOCK_SIZE,1 };


	cudaEvent_t kernel_start;
	cudaEvent_t kernel_end;

	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_end);
	cudaEventRecord(kernel_start);
	cudaEventQuery(kernel_start);
	cuda_transpose << < gird , block  >> > (d_matrix, dtr_matrix, m, n);
	cudaEventRecord(kernel_end);
	cudaEventSynchronize(kernel_end);
	float ms;
	cudaEventElapsedTime(&ms, kernel_start, kernel_end);


	int* hdtr_matrix;
	cudaMallocHost((void**)&hdtr_matrix, sizeof(int) * m * n);
	cudaMemcpy(hdtr_matrix, dtr_matrix, sizeof(int) * m * n, cudaMemcpyDeviceToDevice);
	//print(hdtr_matrix, "GPU", n, m);
	
	if (check(hdtr_matrix, htr_matrix, n, m)) {
		cout << "pass\n";
	}
	else {
		cout << "error\n";
	}
	
	printf("GPU time is : %f \n", ms);

	cudaFree(hdtr_matrix);
	cudaFree(dtr_matrix);
	cudaFree(matrix);
	cudaFree(htr_matrix);
	cudaFree(d_matrix);
	return 0;
}


void print(int* a, string name,int m,int n) {
	cout << "NAME : " << name << endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			printf("%6d ", a[i * n + j]);
		}
		printf("\n");
	}
}

bool check(int* a, int* b, int m, int n) {
	bool check_flag = true;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (a[i * n + j] != b[i * n + j]) {
				return false;
			}
		}
	}
	return check_flag;
}


