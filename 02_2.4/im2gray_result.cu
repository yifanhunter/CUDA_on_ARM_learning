#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


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
 
int main()
{
    Mat src = imread("1.jpg");

    uchar3 *d_in;
    unsigned char *d_out;
    
    int height = src.rows;
    int width = src.cols;
    Mat grayImg(height, width, CV_8UC1, Scalar(0));
 
    cudaMalloc((void**)&d_in, height * width * sizeof(uchar3));
    cudaMalloc((void**)&d_out, height * width * sizeof(unsigned char));
 
    cudaMemcpy(d_in, src.data, height * width * sizeof(uchar3), cudaMemcpyHostToDevice);
 
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
 
    im2gray<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, height, width);

 
    cudaMemcpy(grayImg.data, d_out, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
 
    imwrite("save.png", grayImg);
    cudaFree(d_in);
    cudaFree(d_out);
 
    return 0;

}