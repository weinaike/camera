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
#include <QMessageBox> 
#include <regex>
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
    
}

struct ImageData {
    int width;
    int height;
    int bitDepth;
};


ImageData parseFileName(const std::string& filename) {
    std::regex re(R"((\d+)_(\d+)_(\d+)bit)");
    std::smatch match;
    ImageData imgData = {0, 0, 0};

    if (std::regex_search(filename, match, re) && match.size() == 4) {
        imgData.width = std::stoi(match[1]);
        imgData.height = std::stoi(match[2]);
        imgData.bitDepth = std::stoi(match[3]);
    }
    return imgData;
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
    mIsRawFile = (fileExtension.toLower() == "raw");
    if(mIsRawFile)
    {
        ImageData info =  parseFileName(mFileName.toStdString());
        if(info.width == 0 || info.height == 0 || info.bitDepth == 0)
        {
            // QT 弹窗提示格式不满足，不是raw文件
            QMessageBox::warning(nullptr, QStringLiteral("Error"), QStringLiteral("File format is not supported, please use the format like 640_512_12bit.raw"));
            return false;
        }
        if(info.bitDepth != 8)
        {
            // QT 弹窗提示格式不满足，不是raw文件
            QMessageBox::warning(nullptr, QStringLiteral("Error"), QStringLiteral("Only 8bit raw file is supported now"));
            return false;
        }
        
        if(mfile != NULL)
        {
            // 关闭之前打开的文件
            qDebug("close prior file, %p\n", mfile);
            fclose(mfile);
            mfile=nullptr;            
        }

        #ifdef _WIN32
            errno_t err;
            err = fopen_s(&mfile,mFileName.toStdString().c_str(),"rb");


            if (err != 0) {
                qDebug("Error opening file\n");
                fclose(mfile);
                mfile=nullptr;
                return false;
            }
            else
            {
                qDebug("load video success\n");
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

        mManufacturer = QStringLiteral("zjv");
        mModel = QStringLiteral("raw video camera simulator");
        mSerial = QStringLiteral("0000");

        MallocAllocator a;

        int byte = (info.bitDepth + 8 - 1) / 8;

        uint width = info.width;
        uint height = info.height;
        uint pitch = width * byte;
        uint sampleSize = info.bitDepth;
        // uint samples = 1000;
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
        mCount = 0;
        mCount++;
        mFPS = 200;

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
        printf("PGMCamera::open w[%d] h[%d] stride[%d] bitsPerChannel[%d] mSurfaceFormat[%d]\n", 
            width, height, pitch, sampleSize, mSurfaceFormat);

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

        mStatistics[CameraStatEnum::statCurrFrameID] = mCount;
        mStatistics[CameraStatEnum::statVideoAllFrames] = mSamples;
        mStatistics[CameraStatEnum::statCurrFps100] = mFPS * 100;
        mState = cstStopped;
        emit stateChanged(cstStopped);


    }
    else
    {
        // QT 弹窗提示格式不满足，不是raw文件
        QMessageBox::warning(nullptr, QStringLiteral("Error"), QStringLiteral("File format is not supported"));
        
    }
    return true;
}

bool PGMCamera::start()
{
    mState = cstStreaming;
    emit stateChanged(cstStreaming);
    mInputBuffer.init();
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
        if(mIsRawFile)
        {
            if (!feof(mfile))
            {
                fread(mInputImage.data.get(), 1, mInputImage.wPitch * mInputImage.h, mfile);
                mCount++;


                mStatistics[CameraStatEnum::statCurrFrameID] = mCount;
                mStatistics[CameraStatEnum::statVideoAllFrames] = mSamples;
                mStatistics[CameraStatEnum::statCurrFps100] = mFPS * 100;
                if (mCount % 100 == 0)
                {
                    printf("read frame idx %d\n",mCount);
                }
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

            if(mCount >= mSamples - 1)
            {
                mCount = 0;
                fseek(mfile, 0, SEEK_SET);
                finish = true;
            }

            mRawProc->wake();
        }

        QElapsedTimer timer;
        timer.start();
        while (timer.elapsed() < 1000/mFPS) {
            // Busy-wait or perform other operations
        }
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
        val = 1000000 / mFPS ;
        return true;

    default:
        break;
    }

    return false;
}

bool PGMCamera::setParameter(cmrCameraParameter param, float val)
{
    if(param  == prmFrameRate)
    {
        QMutexLocker l(&mLock);
        mFPS = val;
        return true;
    }    
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
    mCount = value * (mSamples - 1) / 100;
    mStatistics[CameraStatEnum::statCurrFrameID] = mCount;
    if(mIsRawFile)
    {
        fseek(mfile, mCount * mFrameSize, SEEK_SET);

        //获取当前帧
        fread(mInputImage.data.get(), 1, mInputImage.wPitch * mInputImage.h, mfile);
        mCount++;
        #ifdef USE_CUDA
            cudaMemcpy(mInputBuffer.getBuffer(), mInputImage.data.get(), mInputImage.wPitch * mInputImage.h, cudaMemcpyHostToDevice);
        #else
            memcpy(mInputBuffer.getBuffer(), mInputImage.data.get(), mInputImage.wPitch * mInputImage.h);
        #endif
        mInputBuffer.release();
    }
}
