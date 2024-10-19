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
    connect(&mTimer, &QTimer::timeout, this, &ImageResult::UpdateChart);

    mTimer.setSingleShot(false);
    // update every 40 msec
    mTimer.setInterval(200);

    ui->checkBox_control->setEnabled(false);
    ui->checkBox_label->setEnabled(false);
    ui->lineEdit_label_step->setEnabled(false);


    ui->resultTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->resultTable->setColumnWidth(4,2);
    mModel->setRowCount(4);
    mModel->setColumnCount(2);

    
    QDoubleValidator *validator = new QDoubleValidator(0.1, 100, 2, this);  // 限制输入范围在0到100之间
    ui->lineEdit_length->setValidator(validator);
    ui->lineEdit_thick->setValidator(validator);
    ui->lineEdit_speed->setValidator(validator);

    ui->lineEdit_label_step->setValidator(new QIntValidator(0, 100, this));


    // window qt编译， 要求文件编码格式为utf-8 with BOM
    QStringList horizontalHeaders;
    horizontalHeaders << "预测瞬时值" << "预测平均值（总体）" ;
    mModel->setHorizontalHeaderLabels(horizontalHeaders);

    QStringList verticalHeaders;
    verticalHeaders << "熔透状态" << "熔深(um)" << "成形质量(正面)" << "成形质量(背面)";
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
        ui->resultTable->setColumnWidth(col, 150);
    }

    connect(mModel, &QStandardItemModel::dataChanged, this, &ImageResult::on_data_changed);

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
void ImageResult::setCamera(GPUCameraBase* cameraPtr)
{
    mCamera = cameraPtr; 
    if (mCamera != nullptr)
    {
        // mCamera->GetStatistics(CameraStatEnum::statVideoAllFrames, mFrameMax);
        if(mCamera->devID() < 0)
        {
            ui->checkBox_control->setEnabled(false);
            ui->checkBox_label->setEnabled(true);
            ui->lineEdit_label_step->setEnabled(true);
        }
        else
        {
            ui->checkBox_control->setEnabled(true);
            ui->checkBox_label->setEnabled(false);
            ui->lineEdit_label_step->setEnabled(false);
        }
    }    
}


void ImageResult::setProc(RawProcessor* proc)
{
    mProc = proc;
    if(mProc != nullptr)
    {
        QSignalBlocker b(ui->checkBox_control);
        ui->checkBox_control->setChecked(mProc->isControl());
    }
}

void ImageResult::setStreaming(bool value)
{
    mStream = value;
    qDebug("ImageResult::streaming:%d", value);
    ui->horizontalSlider->setEnabled(false);
    if(value)
    {
        mResults.clear();
        mResults_modified.clear();
        mAvg = {0};
        mAvg_modified = {0};    
    }

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

        qreal x = frame_id;
        if(ui->checkBox_label->isChecked())
        {
            float speed = ui->lineEdit_speed->text().toFloat();
            unsigned long long fps = 1;
            mCamera->GetStatistics(CameraStatEnum::statCurrFps100, fps);
            x = speed * x / fps * 100;
        }

        redLineSeries_depth->append(x, -10000);
        redLineSeries_depth->append(x, 10000);
        redLineSeries_stable->append(x, 0);
        redLineSeries_stable->append(x, 100);

        // 获取 与 frame_id 最近的 weld_result
        WeldResult result = {0};
        int minDifference = std::numeric_limits<int>::max();
        std::vector<WeldResult> results;
        if(ui->checkBox_label->isChecked())
        {
            results = mResults_modified;
        }
        else
        {
            results = mResults ;
        }

        for(auto r : results)
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
        CUDAProcessorBase* proc =  mProc->getCUDAProcessor() ;
        proc->transformToGLBuffer(img->data.get(), proc->GetFrameBuffer(), img->w, img->h, img->surfaceFmt);
        setImage((unsigned char *) proc->GetFrameBuffer(), img->w, img->h, img->w);
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
    std::vector<WeldResult> show;
    if(ui->checkBox_label->isChecked())
    {
        show = mResults_modified;
    }
    else
    {
        show = mResults ;
    }
    int cnt = 0;

    for(auto result : show)
    { 
        cnt++;
        if(show.size() > 100 && cnt < (show.size() - 100) && mStream == true)
        {
            continue;   
        }

        qreal x = result.frame_id;
        if(ui->checkBox_label->isChecked())
        {           
            float speed = ui->lineEdit_speed->text().toFloat();
            unsigned long long fps = 1;
            mCamera->GetStatistics(CameraStatEnum::statCurrFps100, fps);
            x = speed * x / fps * 100;
        }

        status_series->append(x, result.weld_status);
        if(result.weld_depth < -0.1)
        {
            minY_depth = qMin(minY_depth, qreal(result.weld_depth));
            maxY_depth = qMax(maxY_depth, qreal(result.weld_depth));
            depth_series->append(x, result.weld_depth);
        }
        if(result.front_quality > 0.01)
        {
            front_series->append(x, result.front_quality);
            // 更新前面质量评分的最大最小值
            minY_front = qMin(minY_front, qreal(result.front_quality));
            maxY_front = qMax(maxY_front, qreal(result.front_quality));
        }
        
        if(result.back_quality > 0.01)
        {
            // 更新背面质量评分的最大最小值
            minY_back = qMin(minY_back, qreal(result.back_quality));
            maxY_back = qMax(maxY_back, qreal(result.back_quality));
            back_series->append(x, result.back_quality);
        }
            
        
        frame_min = qMin(frame_min, x);
        frame_max = qMax(frame_max, x);
        minY_status = qMin(minY_status, qreal(result.weld_status));
        maxY_status = qMax(maxY_status, qreal(result.weld_status));
    }

    maxY_depth = maxY_depth == minY_depth ? minY_depth + 1 : maxY_depth;
    maxY_status = maxY_status == minY_status ? minY_status + 1 : maxY_status;
    maxY_front = maxY_front == minY_front ? minY_front + 1 : maxY_front;
    maxY_back = maxY_back == minY_back ? minY_back + 1 : maxY_back;

    if(show.size() > 0)
    {
        unsigned long long all = 0;
        mCamera->GetStatistics(CameraStatEnum::statVideoAllFrames, all);

        if(ui->checkBox_label->isChecked())
        {
            float speed = ui->lineEdit_speed->text().toFloat();
            unsigned long long fps = 1;
            mCamera->GetStatistics(CameraStatEnum::statCurrFps100, fps);
            all = speed * all / fps * 100;
        }

        if(mStream)
        {
            depth_chart->axisX()->setRange(frame_min, frame_max);
            stable_chart->axisX()->setRange(frame_min, frame_max);
        }
        else
        {
            depth_chart->axisX()->setRange(0, all);
            stable_chart->axisX()->setRange(0, all);
        }

        QValueAxis *axis_depth =  qobject_cast<QValueAxis *>(depth_chart->axes(Qt::Vertical).at(0));
        axis_depth->setRange(minY_depth, maxY_depth);

        // 设置右侧Y轴（背面质量评分）范围
        QValueAxis *axis_status = qobject_cast<QValueAxis *>(depth_chart->axes(Qt::Vertical).at(1));
        axis_status->setRange(1, 4);

        // 设置X轴范围
        
        // 设置左侧Y轴（正面质量评分）范围
        QValueAxis *axis_front = qobject_cast<QValueAxis *>(stable_chart->axes(Qt::Vertical).at(0));
        axis_front->setRange(0, 100);
        // 设置右侧Y轴（背面质量评分）范围
        QValueAxis *axis_back = qobject_cast<QValueAxis *>(stable_chart->axes(Qt::Vertical).at(1));
        axis_back->setRange(0, 100);
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
    update();
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
    mIsInternalUpdate = true;
    // 填写表格内容
    mModel->item(0, 0)->setText(QString::number(result.weld_status));
    mModel->item(1, 0)->setText(QString::number(result.weld_depth, 'f', 2));
    mModel->item(2, 0)->setText(QString::number(result.front_quality, 'f', 2));
    mModel->item(3, 0)->setText(QString::number(result.back_quality, 'f', 2));

    if(ui->checkBox_label->isChecked())
    {   
        int cnt = 0;
        mAvg_modified = {0};
        int status_profile[4] = {0};
        for(auto r : mResults_modified)
        {
            // 标定平均值计算
            unsigned long long fps = 1;
            mCamera->GetStatistics(CameraStatEnum::statCurrFps100, fps);
            float speed = ui->lineEdit_speed->text().toFloat();            
            float distance = std::abs(r.frame_id - result.frame_id) * speed / fps * 100 * 10; //mm
            if(distance < ui->lineEdit_label_step->text().toFloat()/2)
            {
                status_profile[r.weld_status-1]++;
                if(r.weld_status < 4)
                {
                    cnt++;
                    mAvg_modified.weld_depth += r.weld_depth;
                    mAvg_modified.front_quality += r.front_quality;
                    mAvg_modified.back_quality += r.back_quality;
                }
            }            
        }
        if(cnt > 0)
        {
            mAvg_modified.weld_depth /= cnt;
            mAvg_modified.front_quality /= cnt;
            mAvg_modified.back_quality /= cnt;
        }
        int max = 0;
        int max_index = 0;  
        for(int i = 0; i < 4; i++)
        {
            if(status_profile[i] > max)
            {
                max = status_profile[i];
                max_index = i;
            }
        }
        mAvg_modified.weld_status = max_index + 1;
        mModel->item(0, 1)->setText(QString::number(mAvg_modified.weld_status));
        mModel->item(1, 1)->setText(QString::number(mAvg_modified.weld_depth, 'f', 2));
        mModel->item(2, 1)->setText(QString::number(mAvg_modified.front_quality, 'f', 2));
        mModel->item(3, 1)->setText(QString::number(mAvg_modified.back_quality, 'f', 2));
    }
    else
    {
        mModel->item(0, 1)->setText(QString::number(mAvg.weld_status));
        mModel->item(1, 1)->setText(QString::number(mAvg.weld_depth, 'f', 2));
        mModel->item(2, 1)->setText(QString::number(mAvg.front_quality, 'f', 2));
        mModel->item(3, 1)->setText(QString::number(mAvg.back_quality, 'f', 2));
    }
    mIsInternalUpdate = false;
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

    // 判断lineEdit是否为空
    if(ui->lineEdit_thick->text().isEmpty() || ui->lineEdit_speed->text().isEmpty())
    {
        QMessageBox::warning(this, "Warning", " 保存数据之前， 请输入工艺参数.");
        return;
    }

    float thick = ui->lineEdit_thick->text().toFloat();
    float speed = ui->lineEdit_speed->text().toFloat();
    unsigned long long all = 0;
    mCamera->GetStatistics(CameraStatEnum::statFramesTotal, all);

    unsigned long long fps = 1;
    mCamera->GetStatistics(CameraStatEnum::statCurrFps100, fps);
    double length = speed * all / fps * 100;
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
        // 确认对话框
        QMessageBox::StandardButton reply;
        reply = QMessageBox::question(this, "Save Predict Result", "Do you want to save the predict result in " + saveFileName, QMessageBox::Yes|QMessageBox::No);
        if (reply == QMessageBox::Yes)
        {
            QTextStream out(&file);
            out << "thick, speed, length\n";
            out << thick << ", " << speed << ", " << length << "\n";
            out << "frame_id, weld_status, weld_depth, front_quality, back_quality\n";
            for(auto result : mResults)
            {
                out << result.frame_id << ", " << result.weld_status << ", " << result.weld_depth << ", " << result.front_quality << ", " << result.back_quality << "\n";
            }
            file.close();
            sendStatus("Save predict result success");
        }
    }
    else
    {
        QString saveFileName = mFilename + "_predict.txt";      
        QFile file(saveFileName);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            saveFileName = QFileDialog::getSaveFileName(this, tr("Save Predict Result"), saveFileName , tr("Text Files (*.txt)"));
            file.setFileName(saveFileName);            
            if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
            {
                sendStatus("Open file failed");
                return;
            }
        }
        QTextStream out(&file);
        out << "thick, speed, length\n";
        out << thick << ", " << speed << ", " << length << "\n";
        out << "frame_id, weld_status, weld_depth, front_quality, back_quality\n";
        for(auto result : mResults)
        {
            out << result.frame_id << ", " << result.weld_status << ", " << result.weld_depth << ", " << result.front_quality << ", " << result.back_quality << "\n";
        }
        file.close();
        QMessageBox::information(this, "Save Predict Result", "Save predict result success : " + saveFileName);
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

        // 判断lineEdit是否为空
        if(ui->lineEdit_thick->text().isEmpty() || ui->lineEdit_speed->text().isEmpty())
        {
            QMessageBox::warning(this, "Warning", " 保存数据之前， 请输入工艺参数.");
            return;
        }
        // 确认对话框
        QMessageBox::StandardButton reply;
        reply = QMessageBox::question(this, "Save Label Result", "Do you want to save the label result in " + saveFileName, QMessageBox::Yes|QMessageBox::No);
        if (reply == QMessageBox::Yes)
        {
            float thick = ui->lineEdit_thick->text().toFloat();
            float speed = ui->lineEdit_speed->text().toFloat();
            unsigned long long all = 0;
            unsigned long long fps = 1;
            mCamera->GetStatistics(CameraStatEnum::statFramesTotal, all);
            mCamera->GetStatistics(CameraStatEnum::statCurrFps100, fps);
            double length = speed * all / fps * 100;

            QTextStream out(&file);
            out << "thick, speed, length\n";
            out << thick << ", " << speed << ", " << length << "\n";
            out << "frame_id, weld_status, weld_depth, front_quality, back_quality\n";
            for(auto result : mResults_modified)
            {
                out << result.frame_id << ", " << result.weld_status << ", " << result.weld_depth << ", " << result.front_quality << ", " << result.back_quality << "\n";
            }
            file.close();
            sendStatus("Save label result success");
        }
        
    }
    else
    {
        QString saveFileName = mFilename + "_label.txt";      
        QFile file(saveFileName);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            saveFileName = QFileDialog::getSaveFileName(this, tr("Save Label Result"), saveFileName , tr("Text Files (*.txt)"));
            file.setFileName(saveFileName);            
            if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
            {
                sendStatus("Open file failed");
                return;
            }
        }

        // 判断lineEdit是否为空
        if(ui->lineEdit_thick->text().isEmpty() || ui->lineEdit_speed->text().isEmpty())
        {
            QMessageBox::warning(this, "Warning", " 保存数据之前， 请输入工艺参数.");
            return;
        }

        float thick = ui->lineEdit_thick->text().toFloat();
        float speed = ui->lineEdit_speed->text().toFloat();
        unsigned long long all = 0;
        unsigned long long fps = 1;
        mCamera->GetStatistics(CameraStatEnum::statFramesTotal, all);
        mCamera->GetStatistics(CameraStatEnum::statCurrFps100, fps);
        double length = speed * all / fps * 100;

        QTextStream out(&file);
        out << "thick, speed, length\n";
        out << thick << ", " << speed << ", " << length << "\n";
        out << "frame_id, weld_status, weld_depth, front_quality, back_quality\n";
        for(auto result : mResults_modified)
        {
            out << result.frame_id << ", " << result.weld_status << ", " << result.weld_depth << ", " << result.front_quality << ", " << result.back_quality << "\n";
        }
        file.close();
        QMessageBox::information(this, "Save Label Result", "Save label result success : " + saveFileName);
    }
}

void ImageResult::loadData()
{
    mResults.clear();
    mResults_modified.clear();
    mAvg = {0};
    mAvg_modified = {0};

    if(mCamera->devID() < 0)
    {
        PGMCamera *camera = dynamic_cast<PGMCamera *>(mCamera);
        QString filename = camera->getFileName();
        QFile file(filename + "_predict.txt");
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            sendStatus("Open predict file failed");
            return;
        }
        sendStatus("Open predict file success");
        QTextStream in(&file);
        QString line = in.readLine();
        line = in.readLine(); // 工艺参数标签
        line = in.readLine(); // 工艺参数
        QStringList list = line.split(",");
        float thick = list[0].toFloat();
        float speed = list[1].toFloat();
        float length = list[2].toFloat();
        ui->lineEdit_thick->setText(QString::number(thick));
        ui->lineEdit_speed->setText(QString::number(speed));
        ui->lineEdit_length->setText(QString::number(length));
        line = in.readLine(); // 标签

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
            sendStatus("Open label file failed");
            return;
        }
        sendStatus("Open label file success");
        QTextStream in_label(&file_label);
        line = in_label.readLine(); // 工艺参数标签
        line = in_label.readLine(); // 工艺参数
        list = line.split(",");
        thick = list[0].toFloat();
        speed = list[1].toFloat();
        length = list[2].toFloat();
        ui->lineEdit_thick->setText(QString::number(thick));
        ui->lineEdit_speed->setText(QString::number(speed));
        ui->lineEdit_length->setText(QString::number(length));
        line = in_label.readLine(); // 标签
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

        qDebug() << "mResults.size():" << mResults.size();
        if(ui->checkBox_label->isChecked())
        {
            if(mResults_modified.size()>0)  UpdateTable(mResults_modified[0]);
        }
        else
        {
            if(mResults.size()>0)   UpdateTable(mResults[0]);
        }
        
        UpdateChart();
        ui->horizontalSlider->setEnabled(true);
        ui->checkBox_label->setEnabled(true);
        ui->lineEdit_label_step->setEnabled(true);
    }    

}


void ImageResult::clear()
{
    {
        QMutexLocker locker(&mMutex);
        mResults.clear();
        mResults_modified.clear();
    }
    {
        QSignalBlocker b(ui->checkBox_label);
        ui->checkBox_label->setChecked(false);
    }

    {
        QSignalBlocker b(ui->checkBox_control);
        ui->checkBox_control->setChecked(false);
    }
    
    // ui->lineEdit_thick->clear();
    // ui->lineEdit_speed->clear();
    // ui->lineEdit_length->clear();
    WeldResult res = {0};
    mAvg = res;

    UpdateTable(res);
    mFrameMax = 0;
    UpdateSlider();
    UpdateChart();

}

void ImageResult::on_lineEdit_speed_editingFinished()
{
    double speed = ui->lineEdit_speed->text().toFloat();

    unsigned long long all = 0;
    mCamera->GetStatistics(CameraStatEnum::statFramesTotal, all);
    unsigned long long fps = 0;
    mCamera->GetStatistics(CameraStatEnum::statCurrFps100, fps);
    double length = speed * all / fps * 100;    
    qDebug("speed:%f, length:%f, all:%d , fps:%d", speed, length, all, fps);
    ui->lineEdit_length->setText(QString::number(length, 'f', 2));
}


void ImageResult::on_checkBox_label_stateChanged(int arg1)
{
    if(arg1 == Qt::Checked)
    {
        ui->resultTable->setEditTriggers(QAbstractItemView::EditTriggers(QAbstractItemView::DoubleClicked));
        // ui->resultTable->setSelectionBehavior(QAbstractItemView::SelectItems);
        QStringList horizontalHeaders;
        horizontalHeaders << "标注瞬时值" << "标注平均值(平均范围内)" ;
        mModel->setHorizontalHeaderLabels(horizontalHeaders);

        if(mResults_modified.size()>0)  UpdateTable(mResults_modified[0]);
    }
    else
    {
        ui->resultTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        QStringList horizontalHeaders;
        horizontalHeaders << "预测瞬时值" << "预测平均值（总体）" ;
        mModel->setHorizontalHeaderLabels(horizontalHeaders);
        if(mResults.size()>0)   UpdateTable(mResults[0]);
    }
    UpdateChart();
}


void ImageResult::on_resultTable_doubleClicked(const QModelIndex &index)
{
    if(! ui->checkBox_label->isChecked())
    {
        return;
    }

    int row = index.row();
    int col = index.column();
    qDebug("row:%d, col:%d", row, col);

}

void ImageResult::on_data_changed(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles)
{
    if(ui->checkBox_label->isChecked())
    {
        if (mIsInternalUpdate) {
            // 如果是内部更新引起的数据变化，忽略
            return;
        }

        int row = topLeft.row();
        int col = topLeft.column();
        QString value = mModel->item(row, col)->text();
        bool ok;
        float f = value.toFloat(&ok);
        if(!ok)
        {
            QMessageBox::warning(this, "Warning", "请输入数字");
            return;
        }
        qDebug("row:%d, col:%d, value:%f", row, col, f);

               
        unsigned long long frame_id = 0;
        mCamera->GetStatistics(CameraStatEnum::statCurrFrameID, frame_id);
        
        if(col == 0)
        {
            int minDifference = std::numeric_limits<int>::max();
            WeldResult * presult = nullptr;
            for(int i = 0; i < mResults_modified.size(); i++)
            {          
                int difference = std::abs(mResults_modified[i].frame_id - int(frame_id));
                if (difference < minDifference)
                {
                    minDifference = difference;
                    presult = &mResults_modified[i];
                }
            }
            if(presult == nullptr)
            {
                return;
            }
            if(row == 1) presult->weld_depth = f;
            else if(row == 2) presult->front_quality = f;
            else if(row == 3) presult->back_quality = f;
            else if(row == 0) presult->weld_status = int(f);            
        }
        else if(col == 1)
        {   
            for(auto &r : mResults_modified)
            {
                unsigned long long fps = 1;
                mCamera->GetStatistics(CameraStatEnum::statCurrFps100, fps);
                float speed = ui->lineEdit_speed->text().toFloat();            
                float distance = std::abs(r.frame_id - int(frame_id)) * speed / fps * 100 * 10; //mm
                float scale = ui->lineEdit_label_step->text().toFloat();
                if(distance < scale / 2)
                {                    
                    qDebug("distance:%f, :%d, %d, %f",distance,  r.frame_id, frame_id, scale);
                    if(r.weld_status < 4)
                    {
                        if(row == 1) r.weld_depth = f;
                        else if(row == 2) r.front_quality = f;
                        else if(row == 3) r.back_quality = f;
                        else if(row == 0) r.weld_status = int(f);            
                    }                    
                }            
            }
        }             
    }
}

void ImageResult::on_checkBox_control_stateChanged(int arg1)
{

}


void ImageResult::on_checkBox_control_clicked(bool checked)
{
    QString ip = ui->lineEdit_ip->text();

    if(mProc == nullptr)
    {
        return;
    }
    if(mCamera->devID() >= 0)
    {
        if(checked)
        {
            int ret= mProc->startControl(ip.toStdString().c_str(),0,2);
            if(ret < 0)
            {
                {
                    QSignalBlocker b(ui->checkBox_control);
                    ui->checkBox_control->setChecked(false);
                }                
                QMessageBox::warning(this, "Connection Error", "Failed to connect to PLC.");
                
            }
        }
        else
        {
            mProc->stopControl();
            {
                QSignalBlocker b(ui->checkBox_control);
                ui->checkBox_control->setChecked(false);
            }           
        }
    }
}


void ImageResult::on_checkBox_label_clicked(bool checked)
{
    if(checked)
    {
        // 判断ui->lineEdit_speed 是否为空
        if(ui->lineEdit_speed->text().isEmpty())
        {
            QMessageBox::warning(this, "Warning", " 请先输入速度.");
            ui->checkBox_label->setChecked(false);
            return;
        }
    }
}

