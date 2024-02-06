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

#ifndef CUDAPROCESSORBASE_H
#define CUDAPROCESSORBASE_H

#include "CUDAProcessorOptions.h"
#include <list>
#include <memory>

#include "MallocAllocator.h"


#include <QVector>


#include <QObject>
#include <QVector>
#include <QMap>
#include <QMutex>
#include <QPair>
#include <QThread>
#include <QSharedMemory>
#include <QApplication>
//#include <QDesktopWidget>
#include <QDir>
#include <QFileInfo>
#include <QDateTime>
#include <QElapsedTimer>
#include <QDebug>
#include <iostream>

#include "timing.hpp"
#include "helper_image/helper_ppm.hpp"
#include "Image.h"
// #include "helper_jpeg.hpp"
#include "FrameBuffer.h"

#include "Globals.h"

#ifdef USE_CUDA
    #include <cuda_runtime.h>
    #include <cuda.h>
    #include "gray12torgb8.h"
#endif

class CUDAProcessorBase : public QObject
{
    Q_OBJECT
public:
    CUDAProcessorBase(QObject* parent = nullptr);
    ~CUDAProcessorBase() override;

    virtual fastStatus_t Init(CUDAProcessorOptions & options);
    virtual fastStatus_t InitFailed(const char *errStr, fastStatus_t ret);

    virtual fastStatus_t Transform(GPUImage_t *image, CUDAProcessorOptions& opts);
    virtual fastStatus_t TransformFailed(const char *errStr, fastStatus_t ret, fastGpuTimerHandle_t profileTimer);

    virtual fastStatus_t Close();
    virtual void         freeFilters();

    virtual fastStatus_t exportJPEGData(void* dstPtr, unsigned jpegQuality, unsigned &size);


    fastStatus_t exportRawData(void* dstPtr, unsigned int &w, unsigned int &h, unsigned int &pitch);

    void cudaMemoryInfo(const char *str);
    void clearExifSections();

    virtual void* GetFrameBuffer();

    virtual bool isGrayscale(){return false;}
    QString getLastErrorDescription(){ return mErrString; }
    fastStatus_t getLastError(){ return mLastError; }
    bool isInitialized(){return mInitialised;}
    void setInfo(bool info)
    {
        QMutexLocker lock(&mut);
        this->info = info;
    }

    bool haveCUDAJpeg()
    {
        return (hJpegEncoder != nullptr);
    }


    fastBayerPattern_t       BayerFormat;
    QMutex                   mut;
    QMutex                   mut2;
    QMap<QString, float>     stats;
    QMap<QString, float>     stats2;
    fastExportToHostHandle_t hBitmapExport = nullptr;
    fastLut_16_t             outLut;

protected:
    bool         info = true;

    static const int JPEG_HEADER_SIZE = 1024;
    static const int FRAME_TIME = 2;

    fastSurfaceFormat_t surfaceFmt {};
    bool                mInitialised = false;
    QString             mErrString;
    fastStatus_t        mLastError {};

    //Motion Jpeg stuff
    std::unique_ptr<unsigned char, MallocAllocator> hJpegStream;
    fastJpegEncoderHandle_t hJpegEncoder = nullptr;
    fastJfifInfo_t          jfifInfo{};
    unsigned int            jpegStreamSize;
    void *                  srcBuffer = nullptr;

    //OpenGL stuff, 
    //***********************显示需要，重新实现这部分功能，******************
    void*                      hGLBuffer = nullptr;
    fastExportToDeviceHandle_t hExportToDevice = nullptr;

    template<typename T>
    void InitLut(T & param, unsigned short blackLevel, double scale, const QVector<unsigned short> & linearizationLut = QVector<unsigned short>());

    int fmtCudaMalloc(void **ptr, int maxWidth, int maxHeight, fastSurfaceFormat_t srcSurfaceFmt);
    int fastCopyToGPU(GPUImage_t *image, void *ptr, fastSurfaceFormat_t SurfaceFmt, int imgWidth, int imgHeight, bool Packed);
    int transformToGLBuffer(void *ptr, void* hGLBuffer, int  imgWidth, int imgHeight, fastSurfaceFormat_t SurfaceFmt);
signals:
    void initialized(const QString& info);
    void finished();
    void error();
protected:

    unsigned mWidth = 0;
    unsigned mHeight = 0;
    unsigned mPitch = 0;
    unsigned mBitsPerChannel = 8;
    fastSurfaceFormat_t mSurfaceFmt = FAST_RGB8;
    void *mSrcCpuPtr = nullptr;

};
template<typename T>
void CUDAProcessorBase::InitLut(T & param, unsigned short blackLevel, double scale, const QVector<unsigned short> & linearizationLut)
{
    if(linearizationLut.empty())
    {
        int i = 0;
        for(auto& l : param.lut)
        {
            l = static_cast<unsigned short>(qBound<double>(0, (i - blackLevel)  * scale, 1) * 65535);
            i++;
        }
    }
    else
    {
        auto itr = linearizationLut.begin();
        for(auto & l : param.lut)
        {
            if( itr != linearizationLut.end() )
            {
                l = static_cast<unsigned short>(qBound<double>(0, (*itr - blackLevel)  * scale, 1) * 65535);
                ++itr;
                continue;
            }
            break;
        }
    }
}

Q_DECLARE_TYPEINFO(fastJpegExifSection_t, Q_COMPLEX_TYPE);


#endif // CUDAPROCESSORBASE_H
