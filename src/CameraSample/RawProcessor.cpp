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
    mControlPtr.reset(new AsyncControl(32));
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
    mControlPtr.reset(new AsyncControl(32));
}


RawProcessor::~RawProcessor()
{
    stop();

    if(mPipe.get())
    {
        printf("mPipe.reset();\n");
        mPipe.reset();
        QElapsedTimer tm;
        tm.start();
        while(tm.elapsed() <= 1000){};
    }
    if(mControl)
    {
        mControlPtr->disconnect();
    }
    mControlPtr.reset(nullptr);
    mCUDAThread.quit();
    mCUDAThread.wait(3000);
}

fastStatus_t RawProcessor::init()
{
    if(!mProcessorPtr)
        return FAST_INVALID_VALUE;    
    return mProcessorPtr->Init(mOptions);
}

void RawProcessor::start()
{
    if(!mProcessorPtr || mCamera == nullptr)
        return; 
    if(mPipe)
    {
        std::shared_ptr<ZJVIDEO::ControlData> base_mode = std::make_shared<ZJVIDEO::ControlData>(ZJVIDEO::ZJV_CONTROLTYPE_CLEAR_CACHE);
        mPipe->control(base_mode);
    }
    QTimer::singleShot(0, this, [this](){startWorking();});
}

void RawProcessor::stop()
{
    mWorking = false;
    mWaitCond.wakeAll();

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
    printf("RawProcessor::stop() mWorking = %d\n", mWorking);

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

    if(mControl)
    {
        if(!mControlPtr->get_connect_status())
        {
            mControlPtr->connect(mIP, 0, 2);
            mControlPtr->stop();
            mControlPtr->start();
        }
        if(!mControlPtr->get_status())
        {
            mControlPtr->start();
        }
    }


    float fps = 0;
    mCamera->getParameter(GPUCameraBase::prmFrameRate, fps);
    qDebug("RawProcessor::startWorking() fps = %f\n", fps);
    mWake = false;
    qint64 previous = tm.elapsed();
    int frameCount = 0;
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
            break;
        }
           

        if(!mProcessorPtr || mCamera == nullptr)
        {
            continue;
        }

        GPUImage_t* img = mCamera->getFrameBuffer()->getLastImage();
        if(!img)
        {
            continue;
        }

        if(mInfer && mPipe)
        {      
            std::shared_ptr<ZJVIDEO::FrameData> frame = std::make_shared<ZJVIDEO::FrameData>(img->w, img->h, ZJVIDEO::ZJV_IMAGEFORMAT_GRAY8);
            frame->frame_id =  img->frameID;
            frame->fps = fps;
            frame->camera_id = 0;            
            #if USE_CUDA
                cudaMemcpy(frame->data->mutable_gpu_data(), img->data.get(), frame->data->size(), cudaMemcpyDeviceToDevice);
                mPipe->set_input_data(frame);
            #else
                memcpy(frame->data->mutable_cpu_data(), img->data.get(), frame->data->size());
                mPipe->set_input_data(frame);
            #endif
            while(true)
            {
                std::vector<std::shared_ptr<ZJVIDEO::EventData> > datas;
                datas.clear();
                mPipe->get_output_data(datas);
                if (datas.size() == 0) break; //退出
                   
                // qDebug("frame_id: %d, datas.size(): %d\n", frame->frame_id, datas.size());
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
                                weld_result.status_score = weld->status_score;
                                weld_result.weld_status = weld->weld_status - 1000;
                                if(weld->weld_status == 1004)
                                {
                                    weld_result.weld_depth = 0;
                                    weld_result.front_quality = 0;
                                    weld_result.back_quality = 0;
                                }
                                else
                                {
                                    weld_result.weld_depth = weld->weld_depth;
                                    weld_result.front_quality = weld->front_quality;
                                    weld_result.back_quality = weld->back_quality;
                                }
                                if(weld->weld_status > 1001)
                                {
                                    weld_result.weld_depth = 0;
                                }

                                emit send_result(weld_result);
                                
                                // qDebug("WeldResult: frame_id: %d, camera_id: %d, weld_status: %d, status_score: %f, weld_depth: %f, front_quality: %f, back_quality: %f", 
                                //     weld->frame_id, weld->camera_id, weld->weld_status, weld->status_score, weld->weld_depth, weld->front_quality, weld->back_quality);


                                if(mControl)
                                {
                                    // 将功率值写入 PLC 的寄存器，假设我们使用 DB1.DBW0（根据你的配置） 
                                    mControlPtr->put(weld_result, mDBID);
                                }
                            }                
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
                // emit finished();            
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
            else if(mOptions.Codec == CUDAProcessorOptions::vcRAW)
            {
                // qDebug("vcRAW");
                unsigned int bpc = GetBytesPerChannelFromSurface(mOptions.SurfaceFmt);
                // qDebug("width: %d , height:%d , bpc:%d, all:%d", mOptions.Width, mOptions.Height, bpc, mFileWriterPtr->bufferSize());
                unsigned char* buf = mFileWriterPtr->getBuffer();
                if(buf != nullptr)
                {
                    FileWriterTask* task = new FileWriterTask();
                    task->size = mOptions.Width * mOptions.Height * bpc; 
                    memcpy(buf, mProcessorPtr->GetFrameBuffer(), task->size);                   
                    
                    task->data = buf;
                    mFileWriterPtr->put(task);
                    mFileWriterPtr->wake();
                    mFrameCnt++;
                }
            }
        }
        

        frameCount++;
        if(frameCount % 500 == 0)
        {
            qint64 now = tm.elapsed();
            float avg = (float)(now-previous) / frameCount;
            qDebug("avg time: %f\n", avg);
        }
    }
    if(mControl)
    {
        mControlPtr->stop();
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

QString RawProcessor::startWriting()
{
    if(mCamera == nullptr)
        return ".";
    if(mWriting)
        stopWriting();
    mWriting = false;
    if(QFileInfo(mOutputPath).exists())
    {
        QDir dir;
        if(!dir.mkpath(mOutputPath))
            return ".";
    }

    if(!QFileInfo(mOutputPath).isDir())
        return ".";

    mCodec = mOptions.Codec;
    QString fileName;
    if(mCodec == CUDAProcessorOptions::vcMJPG)
    {
        fileName = QDir::toNativeSeparators(
                    QStringLiteral("%1/%2.avi").
                    arg(mOutputPath).
                    arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMddhhmmss"))));
        AsyncMJPEGWriter* writer = new AsyncMJPEGWriter();
        writer->open(mCamera->width(),
                     mCamera->height(),
                     mCamera->fps(),
                     mCamera->isColor() ? mOptions.JpegSamplingFmt : FAST_JPEG_Y,
                     fileName);
        mFileWriterPtr.reset(writer);
    }
    else if(mCodec == CUDAProcessorOptions::vcRAW)
    {
        fileName = QDir::toNativeSeparators(
                    QStringLiteral("%1/video_%2_%3_%4_%5bit.raw")
                    .arg(mOutputPath)
                    .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMddhhmmss")))
                    .arg(mCamera->width())
                    .arg(mCamera->height())
                    .arg(mCamera->bitsPerPixel())
        );   
        AsyncRawWriter* writer = new AsyncRawWriter();
        writer->open(fileName);
        mFileWriterPtr.reset(writer);
    }
    else
    {
        mFileWriterPtr.reset(new AsyncFileWriter());
    }
    unsigned pitch = 3 *(((mOptions.Width + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT ) * FAST_ALIGNMENT);
    unsigned sz = pitch * mOptions.Height;
    mFileWriterPtr->initBuffers(sz);
        
    mFrameCnt = 0;
    mWriting = true;
    return fileName;
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

    if(mCodec == CUDAProcessorOptions::vcRAW)
    {
        AsyncRawWriter* writer = static_cast<AsyncRawWriter*>(mFileWriterPtr.data());
        writer->close();
    }

    mCodec = CUDAProcessorOptions::vcNone;
}


void RawProcessor::startInfer()
{
    if(mInfer)
        return;

    mInfer = false;

    std::string cfg_file = "../configure/pipeline_welding.json";
    mPipe = ZJVIDEO::PublicPipeline::create(cfg_file);
    if(mPipe.get() == nullptr)
    {
        printf("create  pipeline failed ");
        return ;
    }

    mPipe->init();

    std::shared_ptr<ZJVIDEO::SetLoggerLevelControlData> level = std::make_shared<ZJVIDEO::SetLoggerLevelControlData>();
    level->set_level(ZJVIDEO::ZJV_LOGGER_LEVEL_DEBUG);
    std::shared_ptr<ZJVIDEO::ControlData> base_level = std::dynamic_pointer_cast<ZJVIDEO::ControlData>(level);
    mPipe->control(base_level);

    std::shared_ptr<ZJVIDEO::SetRunModeControlData> mode_control = std::make_shared<ZJVIDEO::SetRunModeControlData>();
    mode_control->set_mode(ZJVIDEO::ZJV_PIPELINE_RUN_MODE_LIVING);
    std::shared_ptr<ZJVIDEO::ControlData> base_mode = std::dynamic_pointer_cast<ZJVIDEO::ControlData>(mode_control);
    mPipe->control(base_mode);

    mPipe->start();
    mInfer = true;
}

void RawProcessor::stopInfer()
{
    mInfer = false;
    if(mPipe)
    {
        mPipe->show_debug_info();
        mPipe->stop();
    }
    mPipe.reset();
}

int RawProcessor::connectPLC(const char * ip, int rack, int slot)
{
    strncpy_s(mIP, ip, sizeof(mIP) - 1);
    // 确保字符串以 null 结尾
    mIP[sizeof(mIP) - 1] = '\0';

    if(mControlPtr == nullptr)
        return -1;
    int ret = mControlPtr->connect(ip, rack, slot);

    if (ret == 0) {
        qDebug() << "Connected to PLC successfully.";
        mControl = true;
    } else {
        qDebug() << "Failed to connect to PLC.";
        mControl = false;
        return -1;
    }

    mControlPtr->start();

    return 0;
    
}

int RawProcessor::disconnectPLC()
{
    mControlPtr->disconnect();
    qDebug() << "Disconnected from PLC.";
    mControl = false;
    return 0;
}
