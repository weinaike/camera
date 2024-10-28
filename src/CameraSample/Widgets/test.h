#ifndef TEST_H
#define TEST_H

#include <QWidget>
#include "GPUCameraBase.h"
#include "snap7.h"
namespace Ui {
class QTestWidget;
}

struct ControlData
{
    int     command_id;     // command id    
    float   power;          // Laser power
    float   de_focus;       // De-focus
    float   speed;          // Welding speed
};


class QTestWidget : public QWidget
{
    Q_OBJECT

public:
    explicit QTestWidget(QWidget *parent = nullptr);
    void setCamera(GPUCameraBase* cameraPtr){mCameraPtr = cameraPtr;};
    ~QTestWidget();

private slots:
    void on_pushButton_trigger_clicked();

    void on_pushButton_send_clicked();

    void on_pushButton_disconnect_clicked();

    void on_pushButton_connect_clicked();

    void on_pushButton_sendarea_clicked();

private:
    Ui::QTestWidget *ui;
    GPUCameraBase* mCameraPtr = nullptr;
    TS7Client * mClient = nullptr;
};

#endif // TEST_H
