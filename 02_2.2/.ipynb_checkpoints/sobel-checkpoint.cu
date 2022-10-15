#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

//GPU实现Sobel边缘检测
//             x0 x1 x2 
//             x3 x4 x5 
//             x6 x7 x8 
__global__ void sobel_gpu(unsigned char* in, unsigned char* out, int imgHeight, int imgWidth)
{

}

//CPU实现Sobel边缘检测
void sobel_cpu(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth)
{
    int Gx = 0;
    int Gy = 0;
    for (int i = 1; i < imgHeight - 1; i++)
    {
        uchar* dataUp = srcImg.ptr<uchar>(i - 1);
        uchar* data = srcImg.ptr<uchar>(i);
        uchar* dataDown = srcImg.ptr<uchar>(i + 1);
        uchar* out = dstImg.ptr<uchar>(i);
        for (int j = 1; j < imgWidth - 1; j++)
        {
            Gx = (dataUp[j - 1] + 2 * data[j - 1] + dataDown[j - 1])-(dataUp[j + 1] + 2 * data[j + 1] + dataDown[j + 1]);
            Gy = (dataUp[j - 1] + 2 * dataUp[j] + dataUp[j + 1]) - (dataDown[j - 1] + 2 * dataDown[j] + dataDown[j + 1]);
            out[j] = (abs(Gx) + abs(Gy)) / 2;
        }
    }
}

int main()
{
    //利用opencv的接口读取图片
    Mat img = imread("1.jpg", 0);
    int imgWidth = img.cols;
    int imgHeight = img.rows;

    //利用opencv的接口对读入的grayImg进行去噪
    Mat gaussImg;
    GaussianBlur(img, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);
    //CPU结果为dst_cpu, GPU结果为dst_gpu
    Mat dst_cpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));
    Mat dst_gpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));


    //调用sobel_cpu处理图像
    sobel_cpu(gaussImg, dst_cpu, imgHeight, imgWidth);

    //申请指针并将它指向GPU空间
    size_t num = imgHeight * imgWidth * sizeof(unsigned char);
    unsigned char* in_gpu;
    unsigned char* out_gpu;
    cudaMalloc((void**)&in_gpu, num);
    cudaMalloc((void**)&out_gpu, num);
    //定义grid和block的维度（形状）
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //将数据从CPU传输到GPU
    cudaMemcpy(in_gpu, img.data, num, cudaMemcpyHostToDevice);
    //调用在GPU上运行的核函数
    sobel_gpu<<<blocksPerGrid,threadsPerBlock>>>(in_gpu, out_gpu, imgHeight, imgWidth);

    //将计算结果传回CPU内存
    cudaMemcpy(dst_gpu.data, out_gpu, num, cudaMemcpyDeviceToHost);
    imwrite("save.png", dst_gpu);
    //显示处理结果, 由于这里的Jupyter模式不支持显示图像, 所以我们就不显示了
    //imshow("gpu", dst_gpu);
    //imshow("cpu", dst_cpu);
    //waitKey(0);
    //释放GPU内存空间
    cudaFree(in_gpu);
    cudaFree(out_gpu);
    return 0;
}
