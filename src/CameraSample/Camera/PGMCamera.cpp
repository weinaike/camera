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

#include "PGMCamera.h"
#include "MainWindow.h"
#include "RawProcessor.h"
#include <QFileInfo>
extern int loadPPM(const char *file, void** data, BaseAllocator *alloc, unsigned int &width, unsigned &wPitch, unsigned int &height, unsigned &bitsPerPixel, unsigned &channels);
using CameraStatEnum = GPUCameraBase::cmrCameraStatistic  ;
PGMCamera::PGMCamera(const QString &fileName,
                     fastBayerPattern_t  pattern,
                     bool isColor) :
    mFileName(fileName)
{
    mPattern = pattern;
    mIsColor = isColor;
    mCameraThread.setObjectName(QStringLiteral("PGMCameraThread"));
    moveToThread(&mCameraThread);
    mCameraThread.start();
    qRegisterMetaType<GPUCameraBase::cmrCameraState>("GPUCameraBase::cmrCameraState");
}




PGMCamera::~PGMCamera()
{
    if(mfile)
    {
        fclose(mfile);
        mfile = nullptr;
    }

    mCameraThread.quit();
    mCameraThread.wait(3000);
}

bool PGMCamera::open(int devID)
{
    mDevID = devID;

    QString fileExtension = QFileInfo(mFileName).suffix();
    isRawFile = (fileExtension.toLower() == "raw");
    qDebug("isRawFile %d\n", isRawFile);

    if(isRawFile)
    {
        qDebug("if(isRawFile), %p\n", mfile);
        if(mfile != NULL)
        {
            fclose(mfile);
            mfile=nullptr;
        }

        #ifdef _WIN32
            errno_t err;
            err = fopen_s(&mfile,mFileName.toStdString().c_str(),"rb");
            qDebug("err:%d, %p",err,mfile);

            if (err != 0) {
                qDebug("Error opening file\n");
                fclose(mfile);
                mfile=nullptr;
                return false;
            }
        #else
            mfile = fopen(mFileName.toStdString().c_str(), "rb");
            if(mfile == NULL)
            {
                qDebug("Error opening file\n");
                fclose(mfile);
                mfile=nullptr;
                return false;
            }
        #endif

        mState = cstClosed;

        mManufacturer = QStringLiteral("Fastvideo");
        mModel = QStringLiteral("raw video camera simulator");
        mSerial = QStringLiteral("0000");

        MallocAllocator a;


        uint width = 640;
        uint height = 512;
        uint pitch = width * 2;
        uint sampleSize = 12;
        uint samples = 1000;
        uint frameSize  = pitch * height ;
        mFrameSize = frameSize;
        unsigned char* bits = (unsigned char* )a.allocate(frameSize);


        fseek(mfile, 0, SEEK_END);
        long fileSize = ftell(mfile);
        fseek(mfile, 0, SEEK_SET);
        mSamples = fileSize / frameSize;

        if(fread(bits, 1, frameSize, mfile) != frameSize)
        {
            printf("read frameSize from mfile failed\n");
        }
        cnt++;
        mFPS = 100;

        if(sampleSize == 8)
        {
            mImageFormat = cif8bpp;
            mSurfaceFormat = FAST_I8;
        }
        else if(sampleSize == 12)
        {
            mImageFormat = cif12bpp;
            mSurfaceFormat = FAST_I12;
        }
        else
        {
            mImageFormat = cif16bpp;
            mSurfaceFormat = FAST_I16;
        }

        mWidth = width;
        mHeight = height;
        mWhite = (1 << sampleSize) - 1;
        mBblack = 0;

        mInputImage.w = width;
        mInputImage.h = height;
        mInputImage.surfaceFmt = mSurfaceFormat;
        mInputImage.wPitch = pitch;
        mInputImage.bitsPerChannel = sampleSize;
        printf("PGMCamera::open w[%d] h[%d] stride[%d] bitsPerChannel[%d] mSurfaceFormat[%d]\n", width, height, pitch, sampleSize, mSurfaceFormat);

        try
        {
            mInputImage.data.reset(static_cast<unsigned char*>(a.allocate(mInputImage.wPitch * mInputImage.h)));
        }
        catch(...)
        {
            printf("mInputImage.data.reset(static_cast<unsigned char*>(a.allocate(mInputImage.wPitch * mInputImage.h)))\n");
            return false;
        }

        memcpy(mInputImage.data.get(), bits, pitch * height);
        unsigned short* ptr = (unsigned short* )bits;
        a.deallocate(bits);

        if(!mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat))
        {
            printf("if(!mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat))\n");
            return false;
        }


        mState = cstStopped;
        emit stateChanged(cstStopped);

        return true;
    }



    mState = cstClosed;

    mManufacturer = QStringLiteral("Fastvideo");
    mModel = QStringLiteral("PGM camera simulator");
    mSerial = QStringLiteral("0000");

    MallocAllocator a;
    unsigned char* bits = nullptr;
    uint width = 0;
    uint height = 0;
    uint pitch = 0;
    uint sampleSize = 0;
    uint samples = 0;
    if(1 != loadPPM(mFileName.toStdString().c_str(),
                    reinterpret_cast<void**>(&bits),
                    &a,
                    width, pitch, height,
                    sampleSize, samples))
        return false;

    if(samples != 1)
        return false;

    mFPS = 60;

    if(sampleSize == 8)
    {
        mImageFormat = cif8bpp;
        mSurfaceFormat = FAST_I8;
    }
    else if(sampleSize == 12)
    {
        mImageFormat = cif12bpp;
        mSurfaceFormat = FAST_I12;
    }
    else
    {
        mImageFormat = cif16bpp;
        mSurfaceFormat = FAST_I16;
    }

    mWidth = width;
    mHeight = height;
    mWhite = (1 << sampleSize) - 1;
    mBblack = 0;

    mInputImage.w = width;
    mInputImage.h = height;
    mInputImage.surfaceFmt = mSurfaceFormat;
    mInputImage.wPitch = pitch;
    mInputImage.bitsPerChannel = sampleSize;
    printf("PGMCamera::open w[%d] h[%d] stride[%d] bitsPerChannel[%d] mSurfaceFormat[%d]\n", width, height, pitch, sampleSize, mSurfaceFormat);

    try
    {
        mInputImage.data.reset(static_cast<unsigned char*>(a.allocate(mInputImage.wPitch * mInputImage.h)));
    }
    catch(...)
    {
        return false;
    }

    memcpy(mInputImage.data.get(), bits, pitch * height);
    unsigned short* ptr = (unsigned short* )bits;
    a.deallocate(bits);

    if(!mInputBuffer.allocate(mWidth, mHeight, mSurfaceFormat))
        return false;

    mState = cstStopped;
    emit stateChanged(cstStopped);
    return true;
}

bool PGMCamera::start()
{
    mState = cstStreaming;
    emit stateChanged(cstStreaming);
    QTimer::singleShot(0, this, [this](){startStreaming();});
    return true;
}

bool PGMCamera::stop()
{
    mState = cstStopped;
    emit stateChanged(cstStopped);

    return true;
}

void PGMCamera::close()
{
    stop();
    mState = cstClosed;
    emit stateChanged(cstClosed);
}

void PGMCamera::startStreaming()
{
    if(mState != cstStreaming)
        return;

    if(!mInputImage.data)
        return;

    bool finish = false;

    while(mState == cstStreaming && (finish == false))
    {
        if(isRawFile)
        {
            if (!feof(mfile))
            {
                fread(mInputImage.data.get(), 1, mInputImage.wPitch * mInputImage.h, mfile);
                cnt++;

                mStatistics[CameraStatEnum::statCurrFps100] = mFPS * 100;
                mStatistics[CameraStatEnum::statCurrFrameID] = cnt;
                mStatistics[CameraStatEnum::statVideoAllFrames] = mSamples;
                // printf("read frame idx %d\n",cnt);
                // enum class cmrCameraStatistic {
                //     statFramesTotal = 0, /// Total number of frames acquired
                //     statFramesDropped ,  /// Number of dropped frames
                //     statFramesIncomplete , /// Number of incomplete frames
                //     statCurrFrameID,    /// Current Frame ID (blockID)
                //     statCurrTimestamp,  /// Current Frame Timestamp
                //     statCurrTroughputMbs100, /// Average Thoughtput in Megabits per 100 seconds
                //     statCurrFps100 /// FPS multiplied by 100
                // } ;

            }
            mStatistics[CameraStatEnum::statFramesTotal]++;

        }
    #ifdef USE_CUDA
        cudaMemcpy(mInputBuffer.getBuffer(), mInputImage.data.get(), mInputImage.wPitch * mInputImage.h, cudaMemcpyHostToDevice);
    #else
        memcpy(mInputBuffer.getBuffer(), mInputImage.data.get(), mInputImage.wPitch * mInputImage.h);
    #endif
        mInputBuffer.release();

        {
            QMutexLocker l(&mLock);

            if(cnt >= mSamples - 1)
            {
                cnt = 0;
                fseek(mfile, 0, SEEK_SET);
                finish = true;
            }

            mRawProc->wake();
        }

        QThread::msleep(1000 / mFPS);
    }

    stop();
}
bool PGMCamera::getParameter(cmrCameraParameter param, float& val)
{
    if(param < 0 || param > prmLast)
        return false;

    switch (param)
    {
    case prmFrameRate:
        val = mFPS;
        return true;

    case prmExposureTime:
        val = 1000 / mFPS;
        return true;

    default:
        break;
    }

    return false;
}

bool PGMCamera::setParameter(cmrCameraParameter param, float val)
{
    Q_UNUSED(param)
    Q_UNUSED(val)
    return false;
}

bool PGMCamera::getParameterInfo(cmrParameterInfo& info)
{
    Q_UNUSED(info)
    return false;
}

void PGMCamera::setValue(int value)
{
    QMutexLocker l(&mLock);
    qDebug("PGMCamera:%d",value);
    cnt = value * (mSamples - 1) / 100;
    if(isRawFile)
    {
        fseek(mfile, cnt * mFrameSize, SEEK_SET);

        //获取当前帧
        fread(mInputImage.data.get(), 1, mInputImage.wPitch * mInputImage.h, mfile);
        cnt++;
        #ifdef USE_CUDA
            cudaMemcpy(mInputBuffer.getBuffer(), mInputImage.data.get(), mInputImage.wPitch * mInputImage.h, cudaMemcpyHostToDevice);
        #else
            memcpy(mInputBuffer.getBuffer(), mInputImage.data.get(), mInputImage.wPitch * mInputImage.h);
        #endif
        mInputBuffer.release();
    }
}
