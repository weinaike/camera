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

#include "CUDAProcessorBase.h"
#include <QElapsedTimer>
#include "fastvideo_sdk_define.h"
#include "JpegEncoder.h"

#include <fstream>


CUDAProcessorBase::CUDAProcessorBase(QObject* parent) :
    QObject(parent)
{
    jfifInfo.h_Bytestream = nullptr;
    jfifInfo.exifSections = nullptr;
    jfifInfo.exifSectionsCount = 0;
    jpegStreamSize = 0;
#ifdef USE_CUDA
    size_t freeMem  = 0;
    size_t totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    stats.insert(QStringLiteral("totalMem"),  totalMem);
    stats.insert(QStringLiteral("freeMem"), freeMem);
#endif

}

CUDAProcessorBase::~CUDAProcessorBase()
{
    freeFilters();
}

void CUDAProcessorBase::cudaMemoryInfo(const char *str)
{
    size_t freeMem, totalMem;
#ifdef USE_CUDA
    cudaMemGetInfo(&freeMem, &totalMem);
#endif
    qDebug("%s, free mem\t%zu", str, freeMem);
}

void CUDAProcessorBase::freeFilters()
{
    qDebug("%s\n",__func__);
    if(info)
        qDebug("CUDAProcessorBase::freeFilters");

    Close();

    //OpenGL stuff
   
    mLastError = FAST_OK;

    clearExifSections();

    if(hGLBuffer)
    {
#ifdef ENABLE_GL
        cudaFree( hGLBuffer );
        hGLBuffer  = nullptr;
#else
        free(hGLBuffer);
        hGLBuffer = nullptr;
#endif   
    }
 

    if(srcBuffer)
    {
    #ifdef USE_CUDA
        cudaFree( srcBuffer );
    #else
        free(srcBuffer);
    #endif
        srcBuffer  = nullptr;
    }


    if(mSrcCpuPtr)
    {
        free(mSrcCpuPtr);
        mSrcCpuPtr = nullptr;
    }

}

// 根据宽，高， 以及图像格式， 申请图像的cuda内存
int CUDAProcessorBase::fmtCudaMalloc(void **ptr, int maxWidth, int maxHeight, fastSurfaceFormat_t srcSurfaceFmt)
{

    void * cuda_ptr = nullptr;
#ifdef USE_CUDA
    
    cudaError_t ret ;
    if (FAST_I8 == srcSurfaceFmt)
    {
        ret = cudaMalloc(&cuda_ptr, maxWidth * maxHeight * sizeof(unsigned char));
    }
    else if (FAST_RGB8 == srcSurfaceFmt)
    {
        ret = cudaMalloc(&cuda_ptr, maxWidth * maxHeight * sizeof(unsigned char) * 3 );
    }
    else if (FAST_I16 == srcSurfaceFmt)
    {
        ret = cudaMalloc(&cuda_ptr, maxWidth * maxHeight * sizeof(unsigned short) );
    }
    else if (FAST_RGB16 == srcSurfaceFmt)
    {
        ret = cudaMalloc(&cuda_ptr, maxWidth * maxHeight * sizeof(unsigned short) * 3);
    }
    else if (FAST_I10 == srcSurfaceFmt)
    {
        ret = cudaMalloc(&cuda_ptr, maxWidth * maxHeight * sizeof(unsigned short) );
    }
    else if (FAST_I12 == srcSurfaceFmt)
    {
        ret = cudaMalloc(&cuda_ptr, maxWidth * maxHeight * sizeof(unsigned short) );
    }
    else if (FAST_I14 == srcSurfaceFmt)
    {
        ret = cudaMalloc(&cuda_ptr, maxWidth * maxHeight * sizeof(unsigned short) );
    }
    else if (FAST_BGR8 == srcSurfaceFmt)
    {
        ret = cudaMalloc(&cuda_ptr, maxWidth * maxHeight * sizeof(unsigned char) * 3);
    }
    else if (FAST_RGB12 == srcSurfaceFmt)
    {
        ret = cudaMalloc(&cuda_ptr, maxWidth * maxHeight * sizeof(unsigned short) * 3);
    }
    else if (FAST_BGRX8 == srcSurfaceFmt)
    {
        ret = cudaMalloc(&cuda_ptr, maxWidth * maxHeight * sizeof(unsigned char) * 4);
    }
    else if (FAST_CrCbY8 == srcSurfaceFmt)
    {
        ret = cudaMalloc(&cuda_ptr, maxWidth * maxHeight * sizeof(unsigned char) * 3);
    }
    else if (FAST_YCbCr8 == srcSurfaceFmt)
    {
        ret = cudaMalloc(&cuda_ptr, maxWidth * maxHeight * sizeof(unsigned char) * 3);
    }
    // Add support for other surface formats here
    *ptr = cuda_ptr;
    if (ret != cudaSuccess)
    {
        printf("cudaMalloc srcBuffer failed\n");
        *ptr = nullptr;
        return InitFailed("cudaMalloc failed",FAST_EXECUTION_FAILURE);
    }
    return ret;
#endif
    return 0;
    
}



fastStatus_t CUDAProcessorBase::Init(CUDAProcessorOptions &options)
{
    qDebug("%s\n",__func__);
    fastStatus_t ret = FAST_OK;
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


    stats[QStringLiteral("inputWidth")] = -1;
    stats[QStringLiteral("inputHeight")] = -1;

    fastSurfaceFormat_t srcSurfaceFmt  = options.SurfaceFmt;

    unsigned int width = options.Width;
    unsigned int height = options.Height;

    unsigned int maxWidth = options.MaxWidth;
    unsigned int maxHeight = options.MaxHeight;
    
    if(info)
        qDebug("format = %u", srcSurfaceFmt);

    // 根据maxWidth， maxHeight， 以及srcSurfaceFmt格式， 申请cuda内存    
    fmtCudaMalloc(&srcBuffer, maxWidth, maxHeight, srcSurfaceFmt);

    // malloc srcBuffer
    //bufferPtr = &srcBuffer;

    //Open GL
#ifdef ENABLE_GL
    unsigned maxPitch = 3 * ( ( ( options.MaxWidth + FAST_ALIGNMENT - 1 ) / FAST_ALIGNMENT ) * FAST_ALIGNMENT );
    unsigned bufferSize = maxPitch * options.MaxHeight * sizeof(unsigned char);
    printf("CUDAProcessorBase hGLBuffer:bufferSize %d, w %d, h %d\n", bufferSize, maxWidth, maxHeight);
    if(cudaMalloc( &hGLBuffer, bufferSize ) != cudaSuccess)
    {
        hGLBuffer = nullptr;
        return InitFailed("cudaMalloc failed",ret);
    }
    stats["totalViewportMemory"] = bufferSize;
    cudaMemoryInfo("Created hGLBuffer");

    //JPEG Stuff
    size_t freeMem  = 0;
    size_t totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);

    stats[QStringLiteral("totalMem")] = totalMem;
    stats[QStringLiteral("freeMem")] = freeMem;
#else
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

fastStatus_t CUDAProcessorBase::InitFailed(const char *errStr, fastStatus_t ret)
{
    qDebug("%s\n",__func__);
    mErrString = errStr;
    mLastError = ret;
    mInitialised = false;

    emit error();
    mut.unlock();
    freeFilters();
    qDebug("%s\n",__func__);
    return ret;
}

int CUDAProcessorBase::fastCopyToGPU(GPUImage_t *image, void *dstptr, fastSurfaceFormat_t SurfaceFmt, int imgWidth, int imgHeight, bool Packed)
{
    //将 GPUImage_t 中的数据，拷贝到GPU内存 srcBuffer 中
    //cuda copy
#ifdef USE_CUDA
    cudaError_t ret ;
    if (FAST_I8 == SurfaceFmt)
    {
        ret = cudaMemcpy(dstptr, image->data.get(), imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
    }
    else if (FAST_RGB8 == SurfaceFmt)
    {
        ret = cudaMemcpy(dstptr, image->data.get(), imgWidth * imgHeight * sizeof(unsigned char) * 3, cudaMemcpyDeviceToDevice);
    }
    else if (FAST_I16 == SurfaceFmt)
    {
        ret = cudaMemcpy(dstptr, image->data.get(), imgWidth * imgHeight * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
    }
    else if (FAST_RGB16 == SurfaceFmt)
    {
        ret = cudaMemcpy(dstptr, image->data.get(), imgWidth * imgHeight * sizeof(unsigned short) * 3, cudaMemcpyDeviceToDevice);
    }
    else if (FAST_I10 == SurfaceFmt)
    {
        ret = cudaMemcpy(dstptr, image->data.get(), imgWidth * imgHeight * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
    }
    else if (FAST_I12 == SurfaceFmt)
    {
        ret = cudaMemcpy(dstptr, image->data.get(), imgWidth * imgHeight * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
        
    }
    else if (FAST_I14 == SurfaceFmt)
    {
        ret = cudaMemcpy(dstptr, image->data.get(), imgWidth * imgHeight * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
    }
    else if (FAST_BGR8 == SurfaceFmt)
    {
        ret = cudaMemcpy(dstptr, image->data.get(), imgWidth * imgHeight * sizeof(unsigned char) * 3, cudaMemcpyDeviceToDevice);
    }
    else if (FAST_RGB12 == SurfaceFmt)
    {
        ret = cudaMemcpy(dstptr, image->data.get(), imgWidth * imgHeight * sizeof(unsigned short) * 3, cudaMemcpyDeviceToDevice);
    }
    else if (FAST_BGRX8 == SurfaceFmt)
    {
        ret = cudaMemcpy(dstptr, image->data.get(), imgWidth * imgHeight * sizeof(unsigned char) * 4, cudaMemcpyDeviceToDevice);
    }
    else if (FAST_CrCbY8 == SurfaceFmt)
    {
        ret = cudaMemcpy(dstptr, image->data.get(), imgWidth * imgHeight * sizeof(unsigned char) * 3, cudaMemcpyDeviceToDevice);
    }
    else if (FAST_YCbCr8 == SurfaceFmt)
    {
        ret = cudaMemcpy(dstptr, image->data.get(), imgWidth * imgHeight * sizeof(unsigned char) * 3, cudaMemcpyDeviceToDevice);
    }
    return ret; // Return the value of ret
#endif
    return 0;    
}


int CUDAProcessorBase::transformToGLBuffer(void *srcptr, void* Buffer, int  imgWidth, int imgHeight, fastSurfaceFormat_t SurfaceFmt)
{   
    if (FAST_RGB8 == SurfaceFmt) {
        // Transformation logic for converting from CUDA FAST_RGB8 to OpenGL RGB8
        // copy direct
        #ifdef ENABLE_GL
            cudaMemcpy(Buffer, srcptr, imgWidth * imgHeight * sizeof(unsigned char) * 3, cudaMemcpyDeviceToDevice);
        #elif defined(USE_CUDA)
            cudaMemcpy(Buffer, srcptr, imgWidth * imgHeight * sizeof(unsigned char) * 3, cudaMemcpyDeviceToHost);
        #else
            memcpy(Buffer, srcptr, imgWidth * imgHeight * 3);
        #endif
    }
    else if (FAST_I8 == SurfaceFmt) {
        // Transformation logic for converting from CUDA FAST_I8 to OpenGL RGB8
        #ifdef ENABLE_GL
            // 需要将8bit的灰度图转换为RGB8格式
            convert8BitGrayToRgb(srcptr, Buffer, imgWidth, imgHeight);     
        #elif defined(USE_CUDA)
            // ImageResult显示， 8bit转8bit ，灰度显示
            cudaMemcpy(Buffer, srcptr, imgWidth * imgHeight, cudaMemcpyDeviceToHost);                  
        #else
            // ImageResult显示， 8bit转8bit ，灰度显示
            memcpy(Buffer, srcptr, imgWidth * imgHeight);
        #endif
    }
    else if (FAST_I12 == SurfaceFmt) {
        // Transformation logic for converting from CUDA FAST_I12 to OpenGL RGB8
        // Modify this code based on the actual formats of the buffers
        #ifdef ENABLE_GL // CUDA->CUDA
            convert12BitGrayTo8BitRgb(srcptr, Buffer, imgWidth, imgHeight);
        #elif defined(USE_CUDA)
            // ImageResult显示， 12bit转8bit ，灰度显示 cuda->cpu
            unsigned short* src = (unsigned short*) srcptr;
            unsigned short* src_host = (unsigned short*) malloc(imgWidth * imgHeight * sizeof(unsigned short));
            cudaMemcpy(src_host, src, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);

            unsigned char * dst = (unsigned char *) Buffer;
            for(int i = 0; i < imgWidth * imgHeight; i++) 
            {
                dst[i] = src_host[i] >> 4;
            }
            free(src_host);    
        #else   // CPU->CPU
            // ImageResult显示， 12bit转8bit ，灰度显示
            unsigned short* src = (unsigned short*) srcptr;
            unsigned char * dst = (unsigned char *) Buffer;
            for(int i = 0; i < imgWidth * imgHeight; i++) 
            {
                dst[i] = src[i] >> 4;
            }
        #endif        
    }
    else {
        return -1;
    }
    return 0;
}

fastStatus_t CUDAProcessorBase::Transform(GPUImage_t *image, CUDAProcessorOptions &opts)
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

    stats[QStringLiteral("inputWidth")] = imgWidth;
    stats[QStringLiteral("inputHeight")] = imgHeight;

    QElapsedTimer cpuTimer;
    cpuTimer.start();

#ifdef ENABLE_GL
    //copy image to GPU
    fastCopyToGPU(image, srcBuffer, opts.SurfaceFmt, imgWidth, imgHeight, opts.Packed);


    // tranforme srcBuffer to hGLbuffer(RGB8 format)
    transformToGLBuffer(srcBuffer, hGLBuffer, imgWidth, imgHeight, opts.SurfaceFmt);
    

    if(info)
    {
        cudaDeviceSynchronize();
    }

    float mcs = float(cpuTimer.nsecsElapsed()) / 1000000.f;
    stats[QStringLiteral("totalGPUCPUTime")] = mcs;
    stats[QStringLiteral("totalGPUTime")] = fullTime;

#endif


    locker.unlock();

    // to minimize delay in main thread
    mut2.lock();
    stats2 = stats;
    mut2.unlock();

    emit finished();
    return FAST_OK;
}

fastStatus_t CUDAProcessorBase::TransformFailed(const char *errStr, fastStatus_t ret, fastGpuTimerHandle_t profileTimer)
{
    mLastError = ret;
    mErrString = errStr;
    emit error();
    return ret;
}

fastStatus_t CUDAProcessorBase::Close()
{
    QMutexLocker locker(&mut);

#ifdef USE_CUDA
    size_t freeMem  = 0;
    size_t totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);

    stats[QStringLiteral("totalMem")] = totalMem;
    stats[QStringLiteral("freeMem")] = freeMem;
#endif
    return FAST_OK;
}

fastStatus_t CUDAProcessorBase::exportRawData(void* dstPtr, unsigned int &w, unsigned int &h, unsigned int &pitch)
{

    unsigned bpc = GetBytesPerChannelFromSurface(mSurfaceFmt);
    w = mWidth;
    h = mHeight;
    pitch = mPitch;

    if(dstPtr == nullptr)
        return FAST_OK;

#ifdef USE_CUDA
    cudaMemcpy(dstPtr, srcBuffer, w * h * bpc, cudaMemcpyDeviceToHost);
#else
    if((dstPtr != NULL) && (srcBuffer != NULL))
    {
        memcpy(dstPtr, srcBuffer, w * h * bpc);
    }
    
#endif

    // unsigned short*  dstGrayBuffer = (unsigned short*)malloc(mWidth * mHeight * sizeof(unsigned short));
    // cudaMemcpy(dstGrayBuffer, srcBuffer, mWidth * mHeight * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    // printf("CUDAProcessorBase exportRawData: %d, w :%d, h: %d \n", dstGrayBuffer[0], mWidth, mHeight);
    // free(dstGrayBuffer);

    unsigned short * ptr = (unsigned short *)dstPtr;
    printf("exportRawData:%d\n", ptr[0]);

    return FAST_OK;
}

fastStatus_t CUDAProcessorBase::exportJPEGData(void* dstPtr, unsigned jpegQuality, unsigned& size)
{

    if(dstPtr == nullptr)
        return FAST_OK;
    jpeg_encoder enc;

    std::vector <unsigned char > output;

    enc.encode((unsigned char*)mSrcCpuPtr, mWidth, mHeight, 1, output, jpegQuality);

    size = output.size();
    memcpy(dstPtr, output.data(), size);


    return FAST_OK;
}



void* CUDAProcessorBase::GetFrameBuffer()
{
    if(!mInitialised)
        return nullptr;
    else
        return hGLBuffer;//纯数据，格式要求RGB24位连续存储。
}

void CUDAProcessorBase::clearExifSections()
{
    if(jfifInfo.exifSections != nullptr)
    {
        for(unsigned i = 0; i < jfifInfo.exifSectionsCount; i++)
        {
            free(jfifInfo.exifSections[i].exifData);
            jfifInfo.exifSections[i].exifData = nullptr;
            qDebug("Destroyed jfifInfo.exifSections[%u]", i);
        }
        free(jfifInfo.exifSections);
        jfifInfo.exifSections = nullptr;
        qDebug("Destroyed jfifInfo.exifSections");
    }
    jfifInfo.exifSectionsCount = 0;
}


