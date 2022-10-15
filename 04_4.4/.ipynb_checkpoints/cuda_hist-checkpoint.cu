#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

#define NUM_BINS 256

//将RGB图像转化成灰度图
//out = 0.3 * R + 0.59 * G + 0.11 * B
__global__ void im2gray(uchar3 *in, unsigned char *out, int height, int width)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
 
    if (x < width && y < height)
    {
        uchar3 rgb = in[y * width + x];
 
        out[y * width + x] = 0.30f * rgb.x + 0.59f * rgb.y + 0.11f * rgb.z;
    }
}

__global__ void histogram_smem_atomics(unsigned char *in, int width, int height, int *out)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * gridDim.x * blockDim.x;
 
    atomicAdd(&out[in[index]], 1);

}




int main()
{
    Mat src = imread("1.jpg");

    uchar3 *d_in;
    unsigned char *d_out;
    int hist[NUM_BINS];
    int *d_hist;
    
    memset(hist, 0, NUM_BINS * sizeof(int));
    
    int height = src.rows;
    int width = src.cols;
    Mat grayImg(height, width, CV_8UC1, Scalar(0));
 
    cudaMalloc((void**)&d_in, height * width * sizeof(uchar3));
    cudaMalloc((void**)&d_out, height * width * sizeof(unsigned char));
    cudaMalloc((void**)&d_hist, NUM_BINS * sizeof(int));
 
    cudaMemcpy(d_in, src.data, height * width * sizeof(uchar3), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_hist, hist, 256 * sizeof(int), cudaMemcpyHostToDevice);
 
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
 
    im2gray<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, height, width);
    
    histogram_smem_atomics<<<blocksPerGrid, threadsPerBlock>>>(d_out, width, height,  d_hist);

 
    cudaMemcpy(grayImg.data, d_out, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    /*for(int i = 0; i<NUM_BINS; i++)
    {
        printf("%d\n", hist[i]);
    }*/
    for(int i = 5000; i>0; i-=200)
    {
        for( int j = 0; j<NUM_BINS/3; j++)
        {
            if(hist[j*2]>=i || hist[j*2+1]>=i)
            {
                printf("*");
            }
            else
            {
                printf(" ");
            }
        }
        printf("\n");
    }
 
    imwrite("save.png", grayImg);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_hist);
 
    return 0;

}