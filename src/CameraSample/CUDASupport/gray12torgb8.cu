
#include "gray12torgb8.h"

__global__ void convert12BitGrayTo8BitRgb_(unsigned short* input, uchar3* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned short pixel12bit = input[y * width + x];
        unsigned char pixel8bit = static_cast<unsigned char>(pixel12bit >> 4);
        output[y * width + x] = make_uchar3(pixel8bit, pixel8bit, pixel8bit);
    }
}

__global__ void convert8BitGrayToRgb_(unsigned char* input, uchar3* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned char pixel8bit = input[y * width + x];
        output[y * width + x] = make_uchar3(pixel8bit, pixel8bit, pixel8bit);
    }
}



// 外部调用的函数
extern "C" void convert12BitGrayTo8BitRgb(const void*srdData, void* dstData, int width, int height)
{
    unsigned short* src =  (unsigned short*) srdData;
    uchar3* dst = (uchar3*)dstData;

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    convert12BitGrayTo8BitRgb_<<<gridSize, blockSize>>> (src, dst, width, height);
}


extern "C" void convert8BitGrayToRgb(const void* srcData, void * dstData, int width, int height)
{
    unsigned char* src =  (unsigned char*) srcData;
    uchar3* dst = (uchar3*)dstData;

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    convert8BitGrayToRgb_<<<gridSize, blockSize>>> (src, dst, width, height);
}