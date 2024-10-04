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
#include <QtCharts/QValueAxis>
#include <QVBoxLayout>
#include <QtMath>
#include <QGraphicsLayout>
#include <QFileDialog>
#include <QJsonDocument>
#include <QJsonObject>

#include "MainWindow.h"
#include "PGMCamera.h"
#include <QMessageBox>

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
    // 滑块设置
    connect(ui->horizontalSlider, &QSlider::valueChanged, this,&ImageResult::set_slider_value);

    ui->horizontalSlider->setTracking(false);
    // Set update timer
    connect(&mTimer, &QTimer::timeout, this, &ImageResult::UpdateSlider);    
    // connect(&mTimer, &QTimer::timeout, this, &ImageResult::UpdateChart);

    mTimer.setSingleShot(false);
    // update every 40 msec
    mTimer.setInterval(200);

    ui->resultTable->setColumnWidth(4,4);
    mModel->setRowCount(4);
    mModel->setColumnCount(4);

    // window qt编译， 要求文件编码格式为utf-8 with BOM
    QStringList horizontalHeaders;
    horizontalHeaders << "预测瞬时值" << "预测平均值" << "标注瞬时值" << "标注平均值" ;
    mModel->setHorizontalHeaderLabels(horizontalHeaders);

    QStringList verticalHeaders;
    verticalHeaders << "熔透状态" << "熔深" << "成形质量(正面)" << "成形质量(背面)";
    mModel->setVerticalHeaderLabels(verticalHeaders);
    ui->resultTable->setModel(mModel);

    // Center align the text in the table cells
    for (int row = 0; row < mModel->rowCount(); row++) {
        for (int col = 0; col < mModel->columnCount(); col++) {
            QModelIndex index = mModel->index(row, col);
            ui->resultTable->model()->setData(index, Qt::AlignCenter, Qt::TextAlignmentRole);
            mModel->setItem(row, col, new QStandardItem(""));
        }
    }

    for (int col = 0; col < mModel->columnCount(); col++) {
        ui->resultTable->setColumnWidth(col, 80);
    }


    QFont font;
    font.setBold(false);

    QPen redPen(Qt::red);
    redPen.setWidth(2);


    status_series =  new QLineSeries();
    depth_series =  new QLineSeries();

    // 创建一个图表对象
    depth_chart = new QChart();
    depth_chart->legend()->hide();
    depth_chart->addSeries(depth_series);
    depth_chart->addSeries(status_series);

    // 创建两个轴
    QValueAxis *axisY_depth = new QValueAxis();
    QValueAxis *axisY_status = new QValueAxis();
    QValueAxis *axisX_1 = new QValueAxis();
    axisX_1->setTitleText("帧数");
    axisY_depth->setTitleText("熔深(um)");
    axisY_status->setTitleText("熔透状态");
    axisX_1->setTitleFont(font);
    axisY_depth->setTitleFont(font);
    axisY_status->setTitleFont(font);
    axisY_depth->setLabelFormat("%d");
    axisY_status->setLabelFormat("%d");
    axisX_1->setLabelFormat("%d");

    // 设置 y 坐标轴颜色与曲线颜色相同
    QPen depthPen = depth_series->pen();
    QPen statusPen = status_series->pen();
    axisY_depth->setLinePen(depthPen);
    axisY_depth->setLabelsColor(depthPen.color());
    axisY_status->setLinePen(statusPen);
    axisY_status->setLabelsColor(statusPen.color());



    depth_chart->addAxis(axisY_depth, Qt::AlignLeft);
    depth_chart->addAxis(axisY_status, Qt::AlignRight);
    depth_chart->addAxis(axisX_1, Qt::AlignBottom);

    depth_chart->setMargins(QMargins(0, 0, 0, 0));
    depth_chart->layout()->setContentsMargins(0, 0, 0, 0);
    depth_chart->setBackgroundRoundness(0);



    // 确保 depth_series 使用默认的 Y 轴
    depth_series->attachAxis(axisX_1);
    depth_series->attachAxis(axisY_depth);
    status_series->attachAxis(axisX_1);
    status_series->attachAxis(axisY_status);


    redLineSeries_depth = new QLineSeries();
    redLineSeries_depth->setPen(redPen);
    depth_chart->addSeries(redLineSeries_depth);
    redLineSeries_depth->attachAxis(axisX_1);
    redLineSeries_depth->attachAxis(axisY_depth);


    // 创建一个图表视图对象
    depth_chartView = new QChartView(depth_chart);
    depth_chartView->setRenderHint(QPainter::Antialiasing);

    QVBoxLayout *layout = new QVBoxLayout(ui->depthContainer);
    layout->addWidget(depth_chartView);


    front_series = new QLineSeries();
    back_series = new QLineSeries();


    // 创建一个图表对象
    stable_chart = new QChart();
    stable_chart->legend()->hide();
    stable_chart->addSeries(front_series);
    stable_chart->addSeries(back_series);

    stable_chart->setMargins(QMargins(0, 0, 0, 0));
    stable_chart->layout()->setContentsMargins(0, 0, 0, 0);
    stable_chart->setBackgroundRoundness(0);

    QValueAxis *axisY_front = new QValueAxis();
    QValueAxis *axisY_back = new QValueAxis();
    QValueAxis *axisX_2 = new QValueAxis();
    stable_chart->addAxis(axisY_front, Qt::AlignLeft);
    stable_chart->addAxis(axisY_back, Qt::AlignRight);
    stable_chart->addAxis(axisX_2, Qt::AlignBottom);


    axisX_2->setTitleText("帧数");
    axisY_front->setTitleText("正面质量评分");
    axisY_back->setTitleText("背面质量评分");
    axisY_front->setTitleFont(font);
    axisY_back->setTitleFont(font);
    axisX_2->setTitleFont(font);
    axisY_front->setLabelFormat("%d");
    axisY_back->setLabelFormat("%d");
    axisX_2->setLabelFormat("%d");

    // 设置 y 坐标轴颜色与曲线颜色相同
    QPen frontPen = front_series->pen();
    QPen backPen = back_series->pen();
    axisY_front->setLinePen(frontPen);
    axisY_front->setLabelsColor(frontPen.color());
    axisY_back->setLinePen(backPen);
    axisY_back->setLabelsColor(backPen.color());


    // 确保 depth_series 使用默认的 Y 轴
    front_series->attachAxis(axisX_2);
    front_series->attachAxis(axisY_front);
    back_series->attachAxis(axisX_2);
    back_series->attachAxis(axisY_back);


    redLineSeries_stable = new QLineSeries();
    redLineSeries_stable->setPen(redPen);
    stable_chart->addSeries(redLineSeries_stable);
    redLineSeries_stable->attachAxis(axisX_2);
    redLineSeries_stable->attachAxis(axisY_front);


    // 创建一个图表视图对象
    stable_chartView = new QChartView(stable_chart);
    stable_chartView->setRenderHint(QPainter::Antialiasing);

    QVBoxLayout *stable_layout = new QVBoxLayout(ui->statusContainer);
    stable_layout->addWidget(stable_chartView);

    depth_chartView->setRenderHint(QPainter::Antialiasing, false);
    stable_chartView->setRenderHint(QPainter::Antialiasing, false);

    mFrameMax = 0;
    mTimer.start();
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

        redLineSeries_depth->clear();
        redLineSeries_stable->clear();

        unsigned long long all = 0;
        mCamera->GetStatistics(CameraStatEnum::statVideoAllFrames, all);

        int frame_id = value * (all - 1 )/ 100 ;

        redLineSeries_depth->append(frame_id, -10000);
        redLineSeries_depth->append(frame_id, 10000);
        redLineSeries_stable->append(frame_id, 0);
        redLineSeries_stable->append(frame_id, 100);

        // 获取 与 frame_id 最近的 weld_result
        WeldResult result = {0};
        int minDifference = std::numeric_limits<int>::max();
        for(auto r : mResults)
        {
            int difference = std::abs(r.frame_id - frame_id);
            if (difference < minDifference)
            {
                minDifference = difference;
                result = r;
            }
        }

        UpdateTable(result);

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


void ImageResult::UpdateChart()
{
    QMutexLocker locker(&mMutex);
    // 更新图表
    status_series->clear();
    depth_series->clear();
    front_series->clear();
    back_series->clear();

    qreal frame_min = std::numeric_limits<qreal>::max();
    qreal frame_max = std::numeric_limits<qreal>::lowest();

    qreal minY_front = std::numeric_limits<qreal>::max();
    qreal maxY_front = std::numeric_limits<qreal>::lowest();
    qreal minY_back = std::numeric_limits<qreal>::max();
    qreal maxY_back = std::numeric_limits<qreal>::lowest();

    qreal minY_depth = std::numeric_limits<qreal>::max();
    qreal maxY_depth = std::numeric_limits<qreal>::lowest();

    qreal minY_status = std::numeric_limits<qreal>::max();
    qreal maxY_status = std::numeric_limits<qreal>::lowest();

    // int last_frame_id = mFrameMax; 
    for(auto result : mResults)
    { 
        status_series->append(result.frame_id, result.weld_status);
        depth_series->append(result.frame_id, result.weld_depth);
        front_series->append(result.frame_id, result.front_quality);
        back_series->append(result.frame_id, result.back_quality);
        
        frame_min = qMin(frame_min, qreal(result.frame_id));
        frame_max = qMax(frame_max, qreal(result.frame_id));

        minY_depth = qMin(minY_depth, qreal(result.weld_depth));
        maxY_depth = qMax(maxY_depth, qreal(result.weld_depth));

        minY_status = qMin(minY_status, qreal(result.weld_status));
        maxY_status = qMax(maxY_status, qreal(result.weld_status));

        // 更新前面质量评分的最大最小值
        minY_front = qMin(minY_front, qreal(result.front_quality));
        maxY_front = qMax(maxY_front, qreal(result.front_quality));

        // 更新背面质量评分的最大最小值
        minY_back = qMin(minY_back, qreal(result.back_quality));
        maxY_back = qMax(maxY_back, qreal(result.back_quality));

    }

    maxY_depth = maxY_depth == minY_depth ? minY_depth + 1 : maxY_depth;
    maxY_status = maxY_status == minY_status ? minY_status + 1 : maxY_status;
    maxY_front = maxY_front == minY_front ? minY_front + 1 : maxY_front;
    maxY_back = maxY_back == minY_back ? minY_back + 1 : maxY_back;

    if(mResults.size() > 0)
    {
        unsigned long long all = 0;
        mCamera->GetStatistics(CameraStatEnum::statVideoAllFrames, all);

        all = all > frame_max ? all : frame_max;

        depth_chart->axisX()->setRange(0, all);
        QValueAxis *axis_depth =  qobject_cast<QValueAxis *>(depth_chart->axes(Qt::Vertical).at(0));
        axis_depth->setRange(minY_depth, maxY_depth);

        // 设置右侧Y轴（背面质量评分）范围
        QValueAxis *axis_status = qobject_cast<QValueAxis *>(depth_chart->axes(Qt::Vertical).at(1));
        axis_status->setRange(minY_status, maxY_status);

        // 设置X轴范围
        stable_chart->axisX()->setRange(0, all);
        // 设置左侧Y轴（正面质量评分）范围
        QValueAxis *axis_front = qobject_cast<QValueAxis *>(stable_chart->axes(Qt::Vertical).at(0));
        axis_front->setRange(minY_front, maxY_front);
        // 设置右侧Y轴（背面质量评分）范围
        QValueAxis *axis_back = qobject_cast<QValueAxis *>(stable_chart->axes(Qt::Vertical).at(1));
        axis_back->setRange(minY_back, maxY_back);
    }

    depth_chartView->update();
    stable_chartView->update();

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
    QMutexLocker locker(&mMutex);
    mResults.push_back(result);

    if (mCamera->devID() < 0)
    {
        // 保存结果
        mResults_modified.push_back(result);
    }
    

    // 更新平均值
    mAvg.weld_depth = (mAvg.weld_depth * (mResults.size() - 1)  + result.weld_depth ) / mResults.size();
    mAvg.weld_status = (mAvg.weld_status * (mResults.size() - 1)  + result.weld_status ) / mResults.size();
    mAvg.front_quality = (mAvg.front_quality * (mResults.size() - 1)  + result.front_quality ) / mResults.size();
    mAvg.back_quality = (mAvg.back_quality * (mResults.size() - 1)  + result.back_quality ) / mResults.size();
    // printf("size: %d, mAvg.weld_depth:%d, result.weld_depth:%d \n", mResults.size(), mAvg.weld_depth, result.weld_depth);
    // if(mResults.size() > 10)
    // {
    //     mResults.erase(mResults.begin());
    // }
    // printf("get_result:     frame_id: %d, camera_id: %d, weld_status: %d, status_score: %f, weld_depth: %d, front_quality: %d, back_quality: %d\n",
        // result.frame_id, result.camera_id, result.weld_status, result.status_score, result.weld_depth, result.front_quality, result.back_quality);
    if (mResults.size() % 20 == 1)
    {
        UpdateTable(result);
    }  

}


void ImageResult::UpdateTable(WeldResult result)
{
    // 填写表格内容
    mModel->item(0, 0)->setText(QString::number(result.weld_status));
    mModel->item(1, 0)->setText(QString::number(result.weld_depth));
    mModel->item(2, 0)->setText(QString::number(result.front_quality));
    mModel->item(3, 0)->setText(QString::number(result.back_quality));

    mModel->item(0, 1)->setText(QString::number(mAvg.weld_status));
    mModel->item(1, 1)->setText(QString::number(mAvg.weld_depth));
    mModel->item(2, 1)->setText(QString::number(mAvg.front_quality));
    mModel->item(3, 1)->setText(QString::number(mAvg.back_quality));
}


int ImageResult::RewriteConfigFile(const QString &filename, const QString &key, const QString &value)
{
    QFile file(filename);
    if(!file.open(QIODevice::ReadOnly))
    {

        qDebug() << "Open file failed";
        return -1;
    }
    qDebug() << filename << "Open file success";
    // 解析json文件
    QByteArray data = file.readAll();
    file.close();
    QJsonDocument doc = QJsonDocument::fromJson(data);
    QJsonObject obj = doc.object();

    if (obj.contains(key)) {

        QJsonObject model = obj[key].toObject();
        qDebug() << "old Model file:" << model["model_file"].toString();
        model["model_file"] = value;
        model["weight_file"] = value;
        qDebug() << "new Model file:" << model["model_file"].toString();
        obj[key] = model;
    }
    else {
        qDebug() << "No key in json file";
    }

    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Cannot open file for writing:" << filename;
        return -1;
    }
    doc.setObject(obj);
    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();
    return 0;
}

void ImageResult::on_load_status_model_clicked()
{
    // 打开文件选择窗口，选择 .onnx 文件
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Model File"), "", tr("Model Files (*.onnx)"));
    if(fileName.isEmpty())
    {
        qDebug() << "No file selected";
        return;
    }
    else
    {
        QDir dir(QCoreApplication::applicationDirPath());
        fileName = dir.relativeFilePath(fileName);
        qDebug() << "Relative Path:" << fileName;
        QString filePath = "../configure/cfg_nodes/laser_welding.json";
        QString key = "model";
        int ret = RewriteConfigFile(filePath, key, fileName);
        if (ret < 0)
        {
            qDebug() << "RewriteConfigFile failed ";
        }
    }
}


void ImageResult::on_load_depth_model_clicked()
{
    // 打开文件选择窗口，选择 .onnx 文件
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Model File"), "", tr("Model Files (*.onnx)"));
    if(fileName.isEmpty())
    {
        qDebug() << "No file selected";
        return;
    }
    else
    {
        QDir dir(QCoreApplication::applicationDirPath());
        fileName = dir.relativeFilePath(fileName);
        qDebug() << "Relative Path:" << fileName;
        QString filePath = "../configure/cfg_nodes/laser_welding_depth.json";
        QString key = "model";
        int ret = RewriteConfigFile(filePath, key, fileName);
        if (ret < 0)
        {
            qDebug() << "RewriteConfigFile failed ";
        }
    }

}


void ImageResult::on_load_quality_model_clicked()
{
    // 打开文件选择窗口，选择 .onnx 文件
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Model File"), "", tr("Model Files (*.onnx)"));
    if(fileName.isEmpty())
    {
        qDebug() << "No file selected";
        return;
    }
    else
    {
        QDir dir(QCoreApplication::applicationDirPath());
        fileName = dir.relativeFilePath(fileName);
        qDebug() << "Relative Path:" << fileName;
        QString filePath = "../configure/cfg_nodes/laser_welding_stable.json";
        QString key = "model";
        int ret = RewriteConfigFile(filePath, key, fileName);
        if (ret < 0)
        {
            qDebug() << "RewriteConfigFile failed ";
        }
    }
}

void ImageResult::on_load_video_clicked()
{   
    emit callParentOpenPMG(false);
    // 重新设置滑块
}


void ImageResult::on_save_predict_result_clicked()
{
    if(mCamera == nullptr)
    {
        sendStatus("Please open a camera first");
        return;
    }
    int state = mCamera->state();
    if (state == GPUCameraBase::cstStreaming)
    {
        sendStatus("Please stop the camera before saving the result");
        return;
    }

    if(mCamera->devID() < 0) 
    {
        // 将 GPUCameraBase 类型转换 为 PGMCamera
        PGMCamera *camera = dynamic_cast<PGMCamera *>(mCamera);
        QString filename = camera->getFileName();
        // QString saveFileName = QFileDialog::getSaveFileName(this, tr("Save Predict Result"), filename, tr("Text Files (*.txt)"));
        // 判断是否'.txt'结尾
        // if(!saveFileName.endsWith(".txt"))
        //     saveFileName = saveFileName + ".txt";
        QString saveFileName = filename + "_predict.txt";

        if(saveFileName.isEmpty())
        {
            sendStatus("No file selected");
            return;
        }
        QFile file(saveFileName);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            sendStatus("Open file failed");
            return;
        }
        QTextStream out(&file);

        out << "frame_id, weld_status, weld_depth, front_quality, back_quality\n";
        for(auto result : mResults)
        {
            out << result.frame_id << ", " << result.weld_status << ", " << result.weld_depth << ", " << result.front_quality << ", " << result.back_quality << "\n";
        }
        file.close();
        sendStatus("Save predict result success");
    }
}


void ImageResult::on_save_label_result_clicked()
{

    if(mCamera == nullptr)
    {
        sendStatus("Please open a camera first");
        return;
    }
    int state = mCamera->state();
    if (state == GPUCameraBase::cstStreaming)
    {
        sendStatus("Please stop the camera before saving the result");
        return;
    }

    if(mCamera->devID() < 0) 
    {
        // 将 GPUCameraBase 类型转换 为 PGMCamera
        PGMCamera *camera = dynamic_cast<PGMCamera *>(mCamera);
        QString filename = camera->getFileName();
        // QString saveFileName = QFileDialog::getSaveFileName(this, tr("Save Predict Result"), filename, tr("Text Files (*.txt)"));
        // 判断是否'.txt'结尾
        // if(!saveFileName.endsWith(".txt"))
        //     saveFileName = saveFileName + ".txt";
        QString saveFileName = filename + "_label.txt";

        if(saveFileName.isEmpty())
        {
            sendStatus("No file selected");
            return;
        }
        QFile file(saveFileName);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            sendStatus("Open file failed");
            return;
        }
        QTextStream out(&file);

        out << "frame_id, weld_status, weld_depth, front_quality, back_quality\n";
        for(auto result : mResults_modified)
        {
            out << result.frame_id << ", " << result.weld_status << ", " << result.weld_depth << ", " << result.front_quality << ", " << result.back_quality << "\n";
        }
        file.close();
        sendStatus("Save label result success");
    }
}

void ImageResult::loadData()
{
    mResults.clear();
    mResults_modified.clear();
    mAvg = {0};
    mAvg_modified = {0};
    mModel->clear();

    if(mCamera->devID() < 0)
    {
        PGMCamera *camera = dynamic_cast<PGMCamera *>(mCamera);
        QString filename = camera->getFileName();
        QFile file(filename + "_predict.txt");
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            qDebug() << "Open file failed";
            return;
        }
        QTextStream in(&file);
        QString line = in.readLine();
        while (!in.atEnd())
        {
            line = in.readLine();
            QStringList list = line.split(",");
            WeldResult result = {0};
            result.frame_id = list[0].toInt();
            result.weld_status = list[1].toInt();
            result.weld_depth = list[2].toFloat();
            result.front_quality = list[3].toFloat();
            result.back_quality = list[4].toFloat();
            mResults.push_back(result);

            mAvg.weld_depth = (mAvg.weld_depth * (mResults.size() - 1)  + result.weld_depth ) / mResults.size();
            mAvg.weld_status = (mAvg.weld_status * (mResults.size() - 1)  + result.weld_status ) / mResults.size();
            mAvg.front_quality = (mAvg.front_quality * (mResults.size() - 1)  + result.front_quality ) / mResults.size();
            mAvg.back_quality = (mAvg.back_quality * (mResults.size() - 1)  + result.back_quality ) / mResults.size();            
        }
        file.close();

        // 读取标注结果
        QFile file_label(filename + "_label.txt");
        if (!file_label.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            qDebug() << "Open file failed";
            return;
        }
        QTextStream in_label(&file_label);
        line = in_label.readLine();
        while (!in_label.atEnd())
        {
            line = in_label.readLine();
            QStringList list = line.split(",");
            WeldResult result = {0};
            result.frame_id = list[0].toInt();
            result.weld_status = list[1].toInt();
            result.weld_depth = list[2].toFloat();
            result.front_quality = list[3].toFloat();
            result.back_quality = list[4].toFloat();
            mResults_modified.push_back(result);

            mAvg_modified.weld_depth = (mAvg_modified.weld_depth * (mResults_modified.size() - 1)  + result.weld_depth ) / mResults_modified.size();
            mAvg_modified.weld_status = (mAvg_modified.weld_status * (mResults_modified.size() - 1)  + result.weld_status ) / mResults_modified.size();
            mAvg_modified.front_quality = (mAvg_modified.front_quality * (mResults_modified.size() - 1)  + result.front_quality ) / mResults_modified.size();
            mAvg_modified.back_quality = (mAvg_modified.back_quality * (mResults_modified.size() - 1)  + result.back_quality ) / mResults_modified.size();            
        }
        file_label.close();
    }    

}


void ImageResult::clear()
{
    {
        QMutexLocker locker(&mMutex);
        mResults.clear();
        mResults_modified.clear();
    }
   
    status_series->clear();
    depth_series->clear();
    front_series->clear();
    back_series->clear();

    WeldResult res = {0};
    UpdateTable(res);
    mFrameMax = 0;

}
