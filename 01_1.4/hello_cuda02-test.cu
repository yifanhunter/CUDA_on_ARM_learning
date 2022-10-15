#include <stdio.h>
#include "hello_from_gpu.cuh"

int main(void)
{
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}