/*
 Copyright 2011-2019 Fastvideo, LLC.
 All rights reserved.

 This file is a part of the GPUCameraSample project
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 3. Any third-party SDKs from that project (XIMEA SDK, Fastvideo SDK, etc.) are licensed on different terms. Please see their corresponding license terms.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 The views and conclusions contained in the software and documentation are those
 of the authors and should not be interpreted as representing official policies,
 either expressed or implied, of the FreeBSD Project.
*/

#include "CUDAProcessorGray.h"

#ifdef STATIC_BUILD
extern "C"  fastStatus_t fastEnableWatermark(bool isEnabled);
#endif
#include "fastvideo_sdk_define.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void saveBMP(const char* filename, unsigned char* rgbbuf, int width, int height) 
{
    BITMAPFILEHEADER bfh;  
    BITMAPINFOHEADER bih;  
    /* 
     * Magic number for file. It does not fit in the header structure due to 
     * alignment requirements, so put it outside 
     * 文件的魔术字，由于对齐的需要，没办法将魔术字作为BITMAPFILEHEADER的成员，所以
     * 这里将魔术字放在BITMAPFILEHEADER开头外面的位置。
     */  
    unsigned short bfType=0x4d42;    //'BM'             
    bfh.bfReserved1 = 0;  
    bfh.bfReserved2 = 0;  
    bfh.bfSize = 2/* 2B魔术字 */+sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER)+width*height*3;  
    bfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);  
  
    bih.biSize = sizeof(BITMAPINFOHEADER); 
 
    printf("filesize = %u header size = %u \n", bfh.bfSize, bfh.bfOffBits);
 
    bih.biWidth = width;  
    bih.biHeight = height;  
    bih.biPlanes = 1;  
    bih.biBitCount = 24;  
    bih.biCompression = 0;  
    bih.biSizeImage = 0;  
    bih.biXPelsPerMeter = 5000;  
    bih.biYPelsPerMeter = 5000;  
    bih.biClrUsed = 0;  
    bih.biClrImportant = 0;  
  
    FILE *file = fopen(filename, "w+");  
    if (!file)  
    {  
        printf("Could not write file\n");  
        return;  
    }  
  
    /*Write headers*/  
    fwrite(&bfType,sizeof(bfType),1,file);  
    fwrite(&bfh,sizeof(bfh),1, file);  
    fwrite(&bih,sizeof(bih),1, file);  
  
    fwrite(rgbbuf,width*height*3,1,file);  
    fclose(file);  
}


CUDAProcessorGray::CUDAProcessorGray(QObject* parent) : CUDAProcessorBase(parent)
{
    hGrayToRGBTransform = nullptr;
    dstGrayBuffer = nullptr;
    hDeviceToHostGrayAdapter = nullptr;
}

CUDAProcessorGray::~CUDAProcessorGray()
{
    freeFilters();
}

void CUDAProcessorGray::freeFilters()
{
    CUDAProcessorBase::freeFilters();
}

fastStatus_t CUDAProcessorGray::Init(CUDAProcessorOptions &options)
{
    fastStatus_t ret;
    MallocAllocator alloc;

    if(mInitialised)
    {
        mInitialised = false;
        freeFilters();
    }

    if(info)
        qDebug("Initialising CUDAProcessorBase...");

    mut.lock();

    mLastError = FAST_OK;
    mErrString = QString();

    if(ret != FAST_OK)
        InitFailed("fastInit failed", ret);

    fastSdkParametersHandle_t handle = nullptr;

    //ret = fastDenoiseLibraryInit(handle);
    stats[QStringLiteral("inputWidth")] = -1;
    stats[QStringLiteral("inputHeight")] = -1;

    fastSurfaceFormat_t srcSurfaceFmt  = options.SurfaceFmt;

    unsigned int maxWidth = options.MaxWidth;
    unsigned int maxHeight = options.MaxHeight;
    QElapsedTimer cpuTimer;
    cpuTimer.start();

    //CudaMalloc srcBuffer
    
    // 根据maxWidth， maxHeight， 以及srcSurfaceFmt格式， 申请cuda内存    
    fmtCudaMalloc(&srcBuffer, maxWidth, maxHeight, srcSurfaceFmt);
    mSrcCpuPtr = malloc(maxWidth * maxHeight * 3 * sizeof(unsigned char));
    //Raw data export

    //Export to host rgb image
 
    //Open GL

    if(info)
        qDebug("format = %u", srcSurfaceFmt);


    unsigned maxPitch = 3 * ( ( ( options.MaxWidth + FAST_ALIGNMENT - 1 ) / FAST_ALIGNMENT ) * FAST_ALIGNMENT );
    unsigned bufferSize = maxPitch * options.MaxHeight * sizeof(unsigned char);
    printf("CUDAProcessorGray hGLBuffer:bufferSize %d, w %d, h %d\n", bufferSize, maxWidth, maxHeight);

#ifdef  USE_CUDA

    if(cudaMalloc( &hGLBuffer, bufferSize ) != cudaSuccess)
    {
        hGLBuffer = nullptr;
        return InitFailed("cudaMalloc failed",ret);
    }

    stats["totalViewportMemory"] = bufferSize;
    cudaMemoryInfo("Created hGLBuffer");



    size_t requestedMemSpace = 0;
    size_t tmp = 0;
 

    size_t freeMem  = 0;
    size_t totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);

    float mcs = float(cpuTimer.nsecsElapsed()) / 1000000.f;
    stats[QStringLiteral("totalMem")] = totalMem;
    stats[QStringLiteral("freeMem")] = freeMem;
    stats[QStringLiteral("allocatedMem")] = requestedMemSpace;
    stats[QStringLiteral("elapsedTime")] = mcs;
#else
    unsigned int width = options.Width;
    unsigned int height = options.Height;
    if (srcSurfaceFmt > 4)
    {
        hGLBuffer = malloc(width * height * 3 * sizeof(unsigned char));
    }
    else
    {
        hGLBuffer = malloc(width * height * sizeof(unsigned char));
    }


#endif

    emit initialized(QString());
    mInitialised = true;

    mut.unlock();

    return FAST_OK;
}

fastStatus_t CUDAProcessorGray::Transform(GPUImage_t *image, CUDAProcessorOptions &opts)
{
    QMutexLocker locker(&mut);

    if(image == nullptr)
    {
        mLastError = FAST_INVALID_VALUE;
        mErrString = QStringLiteral("Got null pointer data");
        return mLastError;
    }

    float fullTime = 0.;
    float elapsedTimeGpu = 0.;

    QElapsedTimer cpuTimer;
    cpuTimer.start();

    if(!mInitialised)
        return mLastError;

    mErrString = QString();
    mLastError = FAST_OK;
    fastGpuTimerHandle_t profileTimer = nullptr;

    fastStatus_t ret = FAST_OK;
    unsigned imgWidth  = image->w;
    unsigned imgHeight = image->h;
    mWidth  = image->w;
    mHeight = image->h;
    mPitch  = image->wPitch;
    mBitsPerChannel = image->bitsPerChannel;
    mSurfaceFmt = image->surfaceFmt;
    if(imgWidth > opts.MaxWidth || imgHeight > opts.MaxHeight )
        return TransformFailed("Unsupported image size",FAST_INVALID_FORMAT,profileTimer);

    
#ifdef USE_CUDA

    //copy image to GPU
    fastCopyToGPU(image, srcBuffer, opts.SurfaceFmt, imgWidth, imgHeight, opts.Packed);
    
    // tranforme srcBuffer to hGLbuffer(RGB8 format)
    transformToGLBuffer(srcBuffer, hGLBuffer, imgWidth, imgHeight, opts.SurfaceFmt);
    cudaMemcpy(mSrcCpuPtr, hGLBuffer, imgWidth * imgHeight * sizeof(unsigned char) * 3, cudaMemcpyDeviceToHost);
    // unsigned char*  dstGrayBuffer = (unsigned char*)malloc(imgWidth * imgHeight * sizeof(unsigned char) * 3);
    // cudaMemcpy(dstGrayBuffer, hGLBuffer, 3 * imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // printf("CUDAProcessorGray dstGrayBuffer: %d\n", dstGrayBuffer[0]);
    // free(dstGrayBuffer);
    // C++
    // saveBMP("output.bmp", (unsigned char*)mSrcCpuPtr, mWidth, mHeight);
    // std::ofstream file("output.raw", std::ios::binary);
    // file.write((char*)mSrcCpuPtr, mWidth * mHeight * sizeof(unsigned char) * 3);
    // file.close();
    if(info)
    {
        cudaDeviceSynchronize();
    }
#else
    transformToGLBuffer(image->data.get(), hGLBuffer, imgWidth, imgHeight, opts.SurfaceFmt);
#endif



    locker.unlock();

    if(profileTimer)
    {
        profileTimer = nullptr;
    }

    mut2.lock();
    stats2 = stats;
    mut2.unlock();

    emit finished();

    return FAST_OK;
}
