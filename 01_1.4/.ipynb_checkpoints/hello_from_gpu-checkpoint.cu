#include <stdio.h>
#include "hello_from_gpu.cuh"

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}