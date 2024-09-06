#include "ImageResult.h"
#include "ui_ImageResult.h"
#include <QSlider>
#include <QDateTime>
#include <QScreen>
#include <QPainter>
#include "FrameBuffer.h"
#include <QTime>

#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QChart>
#include <QVBoxLayout>
#include <QtMath>
QT_CHARTS_USE_NAMESPACE

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
    ui->horizontalSlider->setTracking(false);
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




    QLineSeries *series = new QLineSeries();

    // 创建一个图表对象
    QChart *chart = new QChart();
    chart->legend()->hide();
    chart->addSeries(series);
    chart->createDefaultAxes();
    chart->setTitle("Real-time data curve");

    // 创建一个图表视图对象
    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    QVBoxLayout *layout = new QVBoxLayout(ui->chartContainer);
    layout->addWidget(chartView);

    // 模拟实时数据更新

    connect(&mTimer, &QTimer::timeout, this, [series, chart]() {
        static int x = 0;
        series->append(x, qSin(x / 1000) );
        x++;

        // 自适应调整坐标系
        chart->axisX()->setRange(0, x);
        chart->axisY()->setRange(-1, 1); // 假设 y 值在 -1 到 1 之间
    });









}


ImageResult::~ImageResult()
{
    delete ui;
}

void ImageResult::setStreaming(bool value)
{
    mStream = value;
    ui->horizontalSlider->setEnabled(false);
    if(mCamera != nullptr)
    {
        if (mCamera->devID() < 0)
        {
            // 播放视频文件过程中， 不能手动设置
            // 只有导入视频，并且暂停播放，才可滑倒slider
            if(value == false)
            {
                ui->horizontalSlider->setEnabled(true);
            }
        }
    }
}



void ImageResult::set_slider_value(int value)
{
    qDebug("%s %d",__func__, value);
    if(mCamera != nullptr)
    {
        emit set_video_progress(value);
        QThread::msleep(100);
        // 更新图片
        GPUImage_t* img = mCamera->getFrameBuffer()->getLastImage();

        // 显示
        mProc->transformToGLBuffer(img->data.get(), mProc->GetFrameBuffer(), img->w, img->h, img->surfaceFmt);
        setImage((unsigned char *) mProc->GetFrameBuffer(), img->w, img->h, img->w);

    }
}


void ImageResult::UpdateSlider()
{
    QSignalBlocker b(ui->horizontalSlider);

    if(mCamera != nullptr && mStream == true)
    {
        uint64_t cnt = 0;
        uint64_t all = 0;

        if( (mCamera->devID() < 0) && mCamera->GetStatistics(CameraStatEnum::statCurrFrameID, cnt)
            &&  mCamera->GetStatistics(CameraStatEnum::statVideoAllFrames, all))
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
}



void ImageResult::paintEvent(QPaintEvent *event)
{
     Q_UNUSED(event)
    ui->showImage->setPixmap(*mPixmap);
}


void ImageResult::get_result(WeldResult result)
{
    mResults.push_back(result);
    if(mResults.size() > 10)
    {
        mResults.erase(mResults.begin());
    }
    // printf("get_result:     frame_id: %d, camera_id: %d, weld_status: %d, status_score: %f, weld_depth: %f, front_quality: %f, back_quality: %f\n", 
        // result.frame_id, result.camera_id, result.weld_status, result.status_score, result.weld_depth, result.front_quality, result.back_quality);

    
        // 填写表格内容
    mModel->setItem(0, 0, new QStandardItem(QString::number(result.weld_status)));
    mModel->setItem(0, 1, new QStandardItem(QString::number(result.status_score)));
    mModel->setItem(0, 2, new QStandardItem("0.95"));

    mModel->setItem(1, 0, new QStandardItem(QString::number(result.weld_depth)));
    mModel->setItem(1, 1, new QStandardItem("Prediction2"));
    mModel->setItem(1, 2, new QStandardItem("0.85"));

    mModel->setItem(2, 0, new QStandardItem("TrueValue3"));
    mModel->setItem(2, 1, new QStandardItem(QString::number(result.back_quality)));
    mModel->setItem(2, 2, new QStandardItem(QString::number(result.front_quality)));


}

