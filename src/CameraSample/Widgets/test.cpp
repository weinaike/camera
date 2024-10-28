#include "test.h"
#include "ui_test.h"
#include <QMessageBox>

QTestWidget::QTestWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::QTestWidget)
{
    ui->setupUi(this);
    mClient = new TS7Client();
    ui->pushButton_send->setEnabled(false);
    ui->pushButton_connect->setEnabled(true);
    ui->pushButton_disconnect->setEnabled(false);
}

QTestWidget::~QTestWidget()
{
    delete mClient;
    delete ui;
}



void QTestWidget::on_pushButton_trigger_clicked()
{
    qDebug("trigger software");
    if(mCameraPtr)
    {
        bool ret = mCameraPtr->setParameter(GPUCameraBase::prmTriggerSoftware, 1);
        if(ret == false)
        {
            QMessageBox::warning(this, "Error", "Failed to trigger software");
        }
    }
}



void QTestWidget::on_pushButton_send_clicked()
{
    if(mClient)
    {
        int dbID= ui->lineEdit_db->text().toInt();
        float data = ui->lineEdit_data->text().toFloat();
        
        ControlData task = {0};
        task.command_id = 0;
        task.power = data;
        int ret = mClient->DBWrite(dbID, 0, 4, &data);
        if (ret < 0)
        {
            QMessageBox::warning(this, "Error", "Failed to write data to PLC.");
        }
    }
}


void QTestWidget::on_pushButton_disconnect_clicked()
{
    mClient->Disconnect();
    ui->pushButton_send->setEnabled(false);
    ui->pushButton_connect->setEnabled(true);
}


void QTestWidget::on_pushButton_connect_clicked()
{
    char ip[32] = {0};

    strcpy_s(ip, ui->lineEdit_ip->text().toStdString().c_str());

    int res = mClient->ConnectTo(ip, 0, 2);

    if(res == 0)
    {
        ui->pushButton_send->setEnabled(true);
        ui->pushButton_connect->setEnabled(false);
        ui->pushButton_disconnect->setEnabled(true);
    }
    else
    {
        QMessageBox::warning(this, "Error", "Failed to connect to PLC.");
    }
}


void QTestWidget::on_pushButton_sendarea_clicked()
{
    if(mClient)
    {
        int dbID= ui->lineEdit_db->text().toInt();
        float data = ui->lineEdit_data->text().toFloat();

        ControlData task = {0};
        task.command_id = 0;
        task.power = data;
        int ret = mClient->WriteArea(S7AreaDB, dbID, 0, sizeof(ControlData), S7WLByte, &task);
        if (ret != 0)
        {
            QMessageBox::warning(this, "Error", "Failed to write data to PLC.");
        }
    }
}

