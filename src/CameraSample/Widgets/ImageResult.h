#ifndef IMAGERESULT_H
#define IMAGERESULT_H

#include <QWidget>
#include "GPUCameraBase.h"
#include "CUDAProcessorBase.h"
#include "RawProcessor.h"
#include <QStandardItemModel>
#include "customlabel.h"
#include "WeldData.h"
#include <QtCharts/QLineSeries>
#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QMutex>

QT_CHARTS_USE_NAMESPACE
namespace Ui {
class ImageResult;
}

class ImageResult : public QWidget
{
    Q_OBJECT

public:
    explicit ImageResult(QWidget *parent = nullptr);
    ~ImageResult();
    void setCamera(GPUCameraBase* cameraPtr);
    void setProc(RawProcessor* proc);
    void setStreaming(bool value);
    void clear();
    void loadData();
    void setFilename(const QString& filename){mFilename = filename;};


signals:
    void set_video_progress(int value);
    void callParentOpenPMG(bool isBayer);
    void sendStatus(const QString& status);

public slots:
    // 更新图像
    void setImage(unsigned char * ptr, int w, int h, int step);
    // 手动设置视频位置
    void set_slider_value(int value);
    // 播放过程中，实时更新
    void UpdateSlider();
    // 更新图表
    void UpdateChart();
    // 获取焊接结果
    void get_result(WeldResult result);
    // 更新表格
    void UpdateTable(WeldResult result);


protected:
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;


private slots:
    void on_load_status_model_clicked();

    void on_load_depth_model_clicked();

    void on_load_quality_model_clicked();

    void on_load_video_clicked();

    void on_save_predict_result_clicked();

    void on_save_label_result_clicked();

    void on_lineEdit_speed_editingFinished();

    void on_checkBox_label_stateChanged(int arg1);

    void on_resultTable_doubleClicked(const QModelIndex &index);

    // void on_data_changed(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles);
    void on_data_changed(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles = QVector<int>());

    void on_checkBox_control_stateChanged(int arg1);

    void on_checkBox_control_clicked(bool checked);

    void on_checkBox_label_clicked(bool checked);

private:
    int RewriteConfigFile(const QString &filename, const QString &key, const QString &value);

private:
    Ui::ImageResult *ui;
    GPUCameraBase *mCamera{nullptr};
    RawProcessor * mProc{nullptr};
    QTimer mTimer;
    QPixmap *mPixmap;
    QString mFilename;
    QMutex mMutex;

    bool mStream = false;
    QStandardItemModel *mModel = nullptr;
    std::vector<WeldResult> mResults;
    std::vector<WeldResult> mResults_modified;
    WeldResult mAvg;
    WeldResult mAvg_modified;

    QLineSeries *status_series = nullptr;
    QLineSeries *depth_series = nullptr;
    QChart *depth_chart = nullptr;
    QChartView *depth_chartView = nullptr;

    QLineSeries *front_series = nullptr;
    QLineSeries *back_series = nullptr;
    QChart *stable_chart = nullptr;
    QChartView *stable_chartView = nullptr;

    QLineSeries *redLineSeries_depth = nullptr;
    QLineSeries *redLineSeries_stable = nullptr;

    int mFrameMax;
    bool mIsInternalUpdate = false;
};
#endif // IMAGERESULT_H
