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

#include "RawProcessor.h"
#include "CUDAProcessorBase.h"
#include "CUDAProcessorGray.h"
#include "FrameBuffer.h"
#include "GPUCameraBase.h"
#include "MainWindow.h"

#include "avfilewriter/avfilewriter.h"

#include <QElapsedTimer>
#include <QDateTime>
#include <QDebug>
#include <QPoint>
#include "FrameData.h"

#include <string>

RawProcessor::RawProcessor(GPUCameraBase *camera, GLRenderer *renderer):QObject(nullptr),
    mCamera(camera),
    mRenderer(renderer)
{
    if(camera->isColor())
        mProcessorPtr.reset(new CUDAProcessorBase());
    else
        mProcessorPtr.reset(new CUDAProcessorGray());

    connect(mProcessorPtr.data(), SIGNAL(error()), this, SIGNAL(error()));

    mCUDAThread.setObjectName(QStringLiteral("CUDAThread"));
    moveToThread(&mCUDAThread);
    mCUDAThread.start();
}


RawProcessor::RawProcessor(GPUCameraBase *camera):QObject(nullptr),
    mCamera(camera)
{
    if(camera->isColor())
        mProcessorPtr.reset(new CUDAProcessorBase());
    else
        mProcessorPtr.reset(new CUDAProcessorGray());

    connect(mProcessorPtr.data(), SIGNAL(error()), this, SIGNAL(error()));

    mCUDAThread.setObjectName(QStringLiteral("CUDAThread"));
    moveToThread(&mCUDAThread);
    mCUDAThread.start();

    

    
}


RawProcessor::~RawProcessor()
{
    stop();
    mCUDAThread.quit();
    mCUDAThread.wait(3000);
}

fastStatus_t RawProcessor::init()
{
    printf("RawProcessor::init() start\n");
    std::string cfg_file = "../configure/pipeline_welding.json";
    mPipe = ZJVIDEO::PublicPipeline::create(cfg_file);
    mPipe->init();

    std::shared_ptr<ZJVIDEO::SetLoggerLevelControlData> level = std::make_shared<ZJVIDEO::SetLoggerLevelControlData>();
    level->set_level(ZJVIDEO::ZJV_LOGGER_LEVEL_INFO);
    std::shared_ptr<ZJVIDEO::ControlData> base_level = std::dynamic_pointer_cast<ZJVIDEO::ControlData>(level);
    mPipe->control(base_level);

    if(!mProcessorPtr)
        return FAST_INVALID_VALUE;

    return mProcessorPtr->Init(mOptions);
}

void RawProcessor::start()
{
    mPipe->start();
    if(!mProcessorPtr || mCamera == nullptr)
        return;

    QTimer::singleShot(0, this, [this](){startWorking();});
}

void RawProcessor::stop()
{
    mWorking = false;
    mWaitCond.wakeAll();

    mPipe->stop();

    if(mFileWriterPtr)
    {
        mFileWriterPtr->waitFinish();
        mFileWriterPtr->stop();
    }

    //Wait up to 1 sec until mWorking == false
    QElapsedTimer tm;
    tm.start();
    while(mWorking && tm.elapsed() <= 1000)
    {
        QThread::msleep(100);
    }
}

void RawProcessor::wake()
{
    mWake = true;
    mWaitCond.wakeAll();
}

void RawProcessor::updateOptions(const CUDAProcessorOptions& opts)
{
    if(!mProcessorPtr)
        return;
    QMutexLocker lock(&(mProcessorPtr->mut));
    mOptions = opts;
}

void RawProcessor::startWorking()
{
    mWorking = true;

    qint64 lastTime = 0;
    QElapsedTimer tm;
    tm.start();

    QByteArray buffer;
    buffer.resize(mOptions.Width * mOptions.Height * 4);

    int bpc = GetBitsPerChannelFromSurface(mCamera->surfaceFormat());
    int maxVal = (1 << bpc) - 1;
    QString pgmHeader = QString("P5\n%1 %2\n%3\n").arg(mOptions.Width).arg(mOptions.Height).arg(maxVal);
    qDebug() << "pgmHeader: " << pgmHeader;

    mWake = false;

    while(mWorking)
    {
        if(!mWake)
        {
            mWaitMutex.lock();
            mWaitCond.wait(&mWaitMutex);
            mWaitMutex.unlock();
        }
        mWake = false;
        if(!mWorking)
        {
            printf("mWorking = false");
            break;
        }
           

        if(!mProcessorPtr || mCamera == nullptr)
        {
            printf("(!mProcessorPtr || mCamera == nullptr");
            continue;
        }

        GPUImage_t* img = mCamera->getFrameBuffer()->getLastImage();

        if(mPipe)
        {      
            std::shared_ptr<ZJVIDEO::FrameData> frame = std::make_shared<ZJVIDEO::FrameData>(img->w, img->h, ZJVIDEO::ZJV_IMAGEFORMAT_GRAY8);
            frame->frame_id =  img->frameID;
            #if USE_CUDA
                cudaMemcpy(frame->data->mutable_gpu_data(), img->data.get(), frame->data->size(), cudaMemcpyDeviceToDevice);
                mPipe->set_input_data(frame);
            #else
                memcpy(frame->data->mutable_cpu_data(), img->data.get(), frame->data->size());
                mPipe->set_input_data(frame);
            #endif

            std::vector<std::shared_ptr<ZJVIDEO::EventData> > datas;
            datas.clear();
            mPipe->get_output_data(datas);

            for(const auto & data : datas)
            {
                for(const auto & extra : data->extras)
                {
                    if(extra->data_name == "WeldResult")
                    {
                        std::shared_ptr<const ZJVIDEO::WeldResultData> weld = std::dynamic_pointer_cast<const ZJVIDEO::WeldResultData>(extra);

                        if(weld->is_enable)
                        {
                            WeldResult weld_result = {0} ;
                            weld_result.frame_id = weld->frame_id;
                            weld_result.camera_id = weld->camera_id;
                            weld_result.weld_status = weld->weld_status;
                            weld_result.status_score = weld->status_score;
                            weld_result.weld_depth = weld->weld_depth;
                            weld_result.front_quality = weld->front_quality;
                            weld_result.back_quality = weld->back_quality;

                            emit send_result(weld_result);

                            // printf("WeldResult:     frame_id: %d, camera_id: %d, weld_status: %d, status_score: %f, weld_depth: %f, front_quality: %f, back_quality: %f\n", 
                            //     weld->frame_id, weld->camera_id, weld->weld_status, weld->status_score, weld->weld_depth, weld->front_quality, weld->back_quality);
                        }                
                    }
                }            
            }
        }

        // 显示
        mProcessorPtr->Transform(img, mOptions);

        #ifdef ENABLE_GL
            if(mRenderer)
            {
                qint64 curTime = tm.elapsed();
                /// arm processor cannot show 60 fps
                #ifdef __ARM_ARCH
                const qint64 frameTime = 32;
                if(curTime - lastTime >= frameTime)
                #endif
                {
                    if(mOptions.ShowPicture){
                        mRenderer->loadImage(mProcessorPtr->GetFrameBuffer(), mOptions.Width, mOptions.Height);
                        mRenderer->update();
                    }
                    lastTime = curTime;

                    emit finished();
                }
            }
        #else
            // 在将图片传输给GtGWidget进行实现， 控制显示帧率
            qint64 curTime = tm.elapsed();
            const qint64 frameTime = 1000 / mRenderFps;
            if(curTime - lastTime >= frameTime)
            {
                // to minimize delay in main thread
                unsigned char * GLBuffer = (unsigned char * ) mProcessorPtr->GetFrameBuffer();
                if (mOptions.SurfaceFmt > 4)
                {
                    emit show_image(GLBuffer,  mOptions.Width,  mOptions.Height, mOptions.Width * 3);
                }
                else
                {
                    emit show_image(GLBuffer,  mOptions.Width,  mOptions.Height, mOptions.Width);
                }
                lastTime = curTime;
                emit finished();            
            }
        #endif

        if(mWriting && mFileWriterPtr)
        {
            if(mOptions.Codec == CUDAProcessorOptions::vcJPG ||
               mOptions.Codec == CUDAProcessorOptions::vcMJPG)
            {
                unsigned char* buf = mFileWriterPtr->getBuffer();
                if(buf != nullptr)
                {
                    FileWriterTask* task = new FileWriterTask();
                    task->fileName =  QStringLiteral("%1/%2%3.jpg").arg(mOutputPath,mFilePrefix).arg(mFrameCnt);
                    task->size = mFileWriterPtr->bufferSize();
                    task->data = buf;
                    // 获取处理后的图片数据，如果图片不需要处理，可以直接获取原始图片数据
                    mProcessorPtr->exportJPEGData(task->data, mOptions.JpegQuality, task->size);
                    mFileWriterPtr->put(task);
                    mFileWriterPtr->wake();
                    mFrameCnt++;
                }
            }
            else if(mOptions.Codec == CUDAProcessorOptions::vcPGM)
            {
                unsigned char* buf = mFileWriterPtr->getBuffer();
                if(buf != nullptr)
                {
                    unsigned w = 0;
                    unsigned h = 0;
                    unsigned pitch = 0;
                    mProcessorPtr->exportRawData(nullptr, w, h, pitch);

                    int sz = pgmHeader.size() + pitch * h;

                    FileWriterTask* task = new FileWriterTask();
                    task->fileName =  QStringLiteral("%1/%2%3.pgm").arg(mOutputPath,mFilePrefix).arg(mFrameCnt);
                    task->size = sz;

                    task->data = buf;
                    memcpy(task->data, pgmHeader.toStdString().c_str(), pgmHeader.size());
                    unsigned char* data = task->data + pgmHeader.size();
                    mProcessorPtr->exportRawData((void*)data, w, h, pitch);
                    //
                    //Not 8 bit pgm requires big endian byte order
                    if(img->surfaceFmt != FAST_I8)
                    {
                        unsigned short* data16 = (unsigned short*)data;
                        for(unsigned i = 0; i < w * h; i++)
                        {
                            unsigned short val = *data16;
                            *data16 = (val << 8) | (val >> 8);
                            data16++;
                        }
                    }

                    mFileWriterPtr->put(task);
                    mFileWriterPtr->wake();
                    mFrameCnt++;
                }
            }
        }
    }
    mWorking = false;
}

fastStatus_t RawProcessor::getLastError()
{
    if(mProcessorPtr)
        return mProcessorPtr->getLastError();
    else
        return FAST_OK;
}

QString RawProcessor::getLastErrorDescription()
{
    return  (mProcessorPtr) ? mProcessorPtr->getLastErrorDescription() : QString();
}

QMap<QString, float> RawProcessor::getStats()
{
    QMap<QString, float> ret;
    if(mProcessorPtr)
    {
        {
            // to minimize delay in main thread
            mProcessorPtr->mut2.lock();
            ret = mProcessorPtr->stats2;
            mProcessorPtr->mut2.unlock();
        }

        if(mWriting)
        {
            ret[QStringLiteral("procFrames")] = mFileWriterPtr->getProcessedFrames();
            ret[QStringLiteral("droppedFrames")] = mFileWriterPtr->getDroppedFrames();
            AVFileWriter *obj = dynamic_cast<AVFileWriter*>(mFileWriterPtr.data());
            if(obj)
                ret[QStringLiteral("encoding")] = obj->duration();
        }
        else
        {
            ret[QStringLiteral("procFrames")] = -1;
            ret[QStringLiteral("droppedFrames")] = -1;
        }
        ret[QStringLiteral("acqTime")] = acqTimeNsec;

    }

    return ret;
}

void RawProcessor::startWriting()
{
    if(mCamera == nullptr)
        return;

    mWriting = false;
    if(QFileInfo(mOutputPath).exists())
    {
        QDir dir;
        if(!dir.mkpath(mOutputPath))
            return;
    }

    if(!QFileInfo(mOutputPath).isDir())
        return;

    mCodec = mOptions.Codec;

    if(mCodec == CUDAProcessorOptions::vcMJPG)
    {
        QString fileName = QDir::toNativeSeparators(
                    QStringLiteral("%1/%2.avi").
                    arg(mOutputPath).
                    arg(QDateTime::currentDateTime().toString(QStringLiteral("dd_MM_yyyy_hh_mm_ss"))));
        AsyncMJPEGWriter* writer = new AsyncMJPEGWriter();
        writer->open(mCamera->width(),
                     mCamera->height(),
                     25,
                     mCamera->isColor() ? mOptions.JpegSamplingFmt : FAST_JPEG_Y,
                     fileName);
        mFileWriterPtr.reset(writer);
    }    
    else
        mFileWriterPtr.reset(new AsyncFileWriter());

    unsigned pitch = 3 *(((mOptions.Width + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT ) * FAST_ALIGNMENT);
    unsigned sz = pitch * mOptions.Height;
    mFileWriterPtr->initBuffers(sz);

    mFrameCnt = 0;
    mWriting = true;
}

void RawProcessor::stopWriting()
{
    mWriting = false;
    if(!mFileWriterPtr)
    {
        mCodec = CUDAProcessorOptions::vcNone;
        return;
    }

    if(mCodec == CUDAProcessorOptions::vcMJPG)
    {
        AsyncMJPEGWriter* writer = static_cast<AsyncMJPEGWriter*>(mFileWriterPtr.data());
        writer->close();
    }
    mCodec = CUDAProcessorOptions::vcNone;
}
