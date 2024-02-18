#include "ImageResult.h"
#include "ui_ImageResult.h"
#include <QSlider>
#include <QDateTime>
#include <QScreen>
#include <QPainter>

using CameraStatEnum = GPUCameraBase::cmrCameraStatistic  ;
ImageResult::ImageResult(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ImageResult),
    mTimer(this),
    mPixmap(new QPixmap()),
    mModel(new QStandardItemModel(this))
{
    ui->setupUi(this);
    connect(ui->horizontalSlider, &QSlider::valueChanged, this,&ImageResult::set_slider_value);
    // Set update timer
    connect(&mTimer, &QTimer::timeout, this, &ImageResult::UpdateSlider);
    mTimer.setSingleShot(false);
    // update every 40 msec
    mTimer.setInterval(40);


    ui->resultTable->setColumnWidth(3,3);
    mModel->setRowCount(3);
    mModel->setColumnCount(3);

    QStringList horizontalHeaders;
     horizontalHeaders << "GroudTruth" << "Predict" << "Confidence";
    // horizontalHeaders << "真值" << "预测值" << "可信度";
//    horizontalHeaders << (QString::fromUtf8("真值")) << (QString::fromUtf8("预测值")) << (QString::fromUtf8("可信度"));
    mModel->setHorizontalHeaderLabels(horizontalHeaders);

    QStringList verticalHeaders;
     verticalHeaders  << "Penetration" << "Depth" << "Stable";
    // verticalHeaders << "熔透" << "熔深" << "成型稳定性";
//    verticalHeaders << (QString::fromUtf8("熔透")) << (QString::fromUtf8("熔深")) << (QString::fromUtf8("成型稳定性"));
    mModel->setVerticalHeaderLabels(verticalHeaders);
    ui->resultTable->setModel(mModel);
}


ImageResult::~ImageResult()
{
    delete ui;
}

void ImageResult::setStreaming(bool value)
{
    if(value == true)
    {
        ui->horizontalSlider->setEnabled(false);
    }
    else
    {
        ui->horizontalSlider->setEnabled(true);
    }
    mStream = value;
}



void ImageResult::set_slider_value(int value)
{
    qDebug("%s %d",__func__, value);
    emit set_video_progress(value);
}


void ImageResult::UpdateSlider()
{
    if(mCamera != nullptr && mStream == true)
    {
        uint64_t cnt = 0;
        uint64_t all = 0;
        if( mCamera->GetStatistics(CameraStatEnum::statCurrFrameID, cnt)
            &&  mCamera->GetStatistics(CameraStatEnum::statVideoAllFrames, all) )
        {
            if (all > 0)
            {
                int value = cnt * 100 / all;
//                qDebug("value:%d",value);
                ui->horizontalSlider->setValue(value);
            }
            else
            {
                ui->horizontalSlider->setValue(0);
            }
        }
    }

}

void ImageResult::setImage(unsigned char * ptr, int w, int h, int step)
{
    if(w == step)
    {
        QImage img(ptr, w, h, step, QImage::Format_Grayscale8);
        *mPixmap = QPixmap::fromImage(img);
    }
    else if(step == 3 * w)
    {
        QImage img(ptr, w, h, step, QImage::Format_RGB888);
        *mPixmap = QPixmap::fromImage(img);
    }
    else
    {
        // 输出错误信息
        qDebug("ImageResult::setImage: unsupported format");
    }
//     ui->showImage->update();
}



void ImageResult::paintEvent(QPaintEvent *event)
{
     Q_UNUSED(event)
    // QPainter painter(ui->showImage);
    // painter.drawPixmap(0, 0, *mPixmap);
    ui->showImage->clear();
    ui->showImage->setPixmap(*mPixmap);

}

