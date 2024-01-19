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
    CpuAllocator alloc;

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


    //CudaMalloc srcBuffer
    
    // 根据maxWidth， maxHeight， 以及srcSurfaceFmt格式， 申请cuda内存    
    fmtCudaMalloc(&srcBuffer, maxWidth, maxHeight, srcSurfaceFmt);

    //Raw data export

    //Export to host rgb image
 
    //Open GL

    if(info)
        qDebug("format = %u", srcSurfaceFmt);


    unsigned maxPitch = 3 * ( ( ( options.MaxWidth + FAST_ALIGNMENT - 1 ) / FAST_ALIGNMENT ) * FAST_ALIGNMENT );
    unsigned bufferSize = maxPitch * options.MaxHeight * sizeof(unsigned char);
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

    stats[QStringLiteral("totalMem")] = totalMem;
    stats[QStringLiteral("freeMem")] = freeMem;
    stats[QStringLiteral("allocatedMem")] = requestedMemSpace;

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

    if(imgWidth > opts.MaxWidth || imgHeight > opts.MaxHeight )
        return TransformFailed("Unsupported image size",FAST_INVALID_FORMAT,profileTimer);

    //copy image to GPU
    fastCopyToGPU(image, srcBuffer, opts.SurfaceFmt, imgWidth, imgHeight, opts.Packed);

    // tranforme srcBuffer to hGLbuffer(RGB8 format)
    transformToGLBuffer(srcBuffer, hGLBuffer, imgWidth, imgHeight, opts.SurfaceFmt);
    

    if(info)
    {
        cudaDeviceSynchronize();
    }

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
