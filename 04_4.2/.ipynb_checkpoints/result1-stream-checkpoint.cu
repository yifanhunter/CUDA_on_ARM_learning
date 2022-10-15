#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define N   (1024*1024)
#define FULL_DATA_SIZE   (N*20)


__global__ void kernel( int *a, int *b, int *c ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}


int main( void ) {
    cudaDeviceProp  prop;
    int whichDevice;
    CHECK( cudaGetDevice( &whichDevice ) );
    CHECK( cudaGetDeviceProperties( &prop, whichDevice ) );
    if (!prop.deviceOverlap) {
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
        return 0;
    }

    cudaEvent_t     start, stop;
    float           elapsedTime;

    cudaStream_t    stream0, stream1;
    int *host_a, *host_b, *host_c;
    int *dev_a0, *dev_b0, *dev_c0;
    int *dev_a1, *dev_b1, *dev_c1;

    // start the timers
    CHECK( cudaEventCreate( &start ) );
    CHECK( cudaEventCreate( &stop ) );

    // initialize the streams
    CHECK( cudaStreamCreate( &stream0 ) );
    CHECK( cudaStreamCreate( &stream1 ) );

    // allocate the memory on the GPU
    CHECK( cudaMalloc( (void**)&dev_a0, N * sizeof(int) ) );
    CHECK( cudaMalloc( (void**)&dev_b0, N * sizeof(int) ) );
    CHECK( cudaMalloc( (void**)&dev_c0, N * sizeof(int) ) );
    CHECK( cudaMalloc( (void**)&dev_a1, N * sizeof(int) ) );
    CHECK( cudaMalloc( (void**)&dev_b1, N * sizeof(int) ) );
    CHECK( cudaMalloc( (void**)&dev_c1, N * sizeof(int) ) );

    // allocate host locked memory, used to stream
    CHECK( cudaHostAlloc( (void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault ) );
    CHECK( cudaHostAlloc( (void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault ) );
    CHECK( cudaHostAlloc( (void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault ) );

    for (int i=0; i<FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    CHECK( cudaEventRecord( start, 0 ) );
    // now loop over full data, in bite-sized chunks
    for (int i=0; i<FULL_DATA_SIZE; i+= N*2) {
        // enqueue copies of a in stream0 and stream1
        CHECK( cudaMemcpyAsync( dev_a0, host_a+i, N * sizeof(int), cudaMemcpyHostToDevice, stream0 ) );
        CHECK( cudaMemcpyAsync( dev_a1, host_a+i+N, N * sizeof(int), cudaMemcpyHostToDevice, stream1 ) );
        // enqueue copies of b in stream0 and stream1
        CHECK( cudaMemcpyAsync( dev_b0, host_b+i, N * sizeof(int), cudaMemcpyHostToDevice, stream0 ) );
        CHECK( cudaMemcpyAsync( dev_b1, host_b+i+N, N * sizeof(int), cudaMemcpyHostToDevice, stream1 ) );

        kernel<<<N/256,256,0,stream0>>>( dev_a0, dev_b0, dev_c0 );
        kernel<<<N/256,256,0,stream1>>>( dev_a1, dev_b1, dev_c1 );

        CHECK( cudaMemcpyAsync( host_c+i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0 ) );
        CHECK( cudaMemcpyAsync( host_c+i+N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1 ) );
    }
    CHECK( cudaStreamSynchronize( stream0 ) );
    CHECK( cudaStreamSynchronize( stream1 ) );

    CHECK( cudaEventRecord( stop, 0 ) );

    CHECK( cudaEventSynchronize( stop ) );
    CHECK( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time taken:  %3.1f ms\n", elapsedTime );

    // cleanup the streams and memory
    CHECK( cudaFreeHost( host_a ) );
    CHECK( cudaFreeHost( host_b ) );
    CHECK( cudaFreeHost( host_c ) );
    CHECK( cudaFree( dev_a0 ) );
    CHECK( cudaFree( dev_b0 ) );
    CHECK( cudaFree( dev_c0 ) );
    CHECK( cudaFree( dev_a1 ) );
    CHECK( cudaFree( dev_b1 ) );
    CHECK( cudaFree( dev_c1 ) );
    CHECK( cudaStreamDestroy( stream0 ) );
    CHECK( cudaStreamDestroy( stream1 ) );

    return 0;
}

