#ifndef IMAGERESULT_H
#define IMAGERESULT_H

#include <QWidget>
#include "GPUCameraBase.h"
#include <QStandardItemModel>
namespace Ui {
class ImageResult;
}

class ImageResult : public QWidget
{
    Q_OBJECT

public:
    explicit ImageResult(QWidget *parent = nullptr);
    ~ImageResult();
    void setCamera(GPUCameraBase* cameraPtr){mCamera = cameraPtr;}
    void setStreaming(bool value);
    void start(){mTimer.start();}
    void stop(){mTimer.stop();}



signals:
    void set_video_progress(int value);

public slots:
    // 更新图像
    void setImage(unsigned char * ptr, int w, int h, int step);
    // 手动设置视频位置
    void set_slider_value(int value);
    // 播放过程中，实时更新
    void UpdateSlider();

protected:
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;


private:
    Ui::ImageResult *ui;
    GPUCameraBase *mCamera{nullptr};
    QTimer mTimer;
    QPixmap *mPixmap;
    bool mStream = false;
    QStandardItemModel *mModel = nullptr;
};

#endif // IMAGERESULT_H
