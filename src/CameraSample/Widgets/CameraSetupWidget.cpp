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

#include "CameraSetupWidget.h"
#include "ui_CameraSetupWidget.h"
#include <QFileDialog>
#include <QString>
#include <QMessageBox>
#include <fstream>

CameraSetupWidget::CameraSetupWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::CameraSetupWidget)
{
    ui->setupUi(this);
    ui->spnFrameRate->setEnabled(false);
    ui->spnExposureTime->setEnabled(false);
    ui->radioButton_ext->setCheckable(false);
    ui->radioButton_int->setCheckable(false);

    ui->lineEdit_frames->setValidator(new QIntValidator(0, 1000000, this));
    ui->lineEdit_frames->setEnabled(false);
}

CameraSetupWidget::~CameraSetupWidget()
{
    delete ui;
}

void CameraSetupWidget::setCamera(GPUCameraBase *cameraPtr)
{
    mCameraPtr = cameraPtr;

    if (mCameraPtr != nullptr)
    {
        qDebug("Camera ID: %d\n", cameraPtr->devID());
        if (cameraPtr->devID() < 0)
        {
            ui->radioButton_ext->setCheckable(false);
            ui->radioButton_int->setCheckable(false);
        }
        else
        {
            ui->radioButton_ext->setCheckable(true);
            ui->radioButton_int->setCheckable(true);
            ui->lineEdit_frames->setEnabled(true);
            std::ifstream ifs("config.txt");
            if (ifs)
            {
                std::string str;
                ifs >> str;
                ifs.close();
                ui->lineEdit_frames->setText(QString::fromStdString(str));
            }
            
            float val = 0;
            bool ret = cameraPtr->getParameter(GPUCameraBase::prmTriggerMode, val);
            if (ret == false)
            {
                QMessageBox::warning(this, "Error", "Failed to get trigger mode");
            }
            if (int(val) == 1)
            {
                QSignalBlocker b(ui->radioButton_ext);
                ui->radioButton_ext->setChecked(true);
                emit modeChanged(INPUT_MODE::MODE_CAMERA_EXTERNAL_TRIGGER);
                cameraPtr->setParameter(GPUCameraBase::prmAcqFrameCount, ui->lineEdit_frames->text().toInt());
            }
            else
            {
                QSignalBlocker b(ui->radioButton_int);
                ui->radioButton_int->setChecked(true);
                emit modeChanged(INPUT_MODE::MODE_CAMERA_INTERNAL_TRIGGER);
                ui->lineEdit_frames->setText("0");
                cameraPtr->setParameter(GPUCameraBase::prmAcqFrameCount, 0);
            }

            ui->label_ip->setText(cameraPtr->ipAddress());
            ui->label_mac->setText(cameraPtr->macAddress());
            ui->label_gateway->setText(cameraPtr->gateway());
            ui->label_subnet->setText(cameraPtr->subnetMask());

            std::string str;
            ret = cameraPtr->getParameter(GPUCameraBase::prmTriggerSource, str);
            if (ret == false)
            {
                ui->label_source->setText("Unknown");
            }
            else
            {
                ui->label_source->setText(QString::fromStdString(str));
            }
            ret = cameraPtr->getParameter(GPUCameraBase::prmTriggerSelector, str);
            if (ret == false)
            {
                ui->label_selector->setText("Unknown");
            }
            else
            {
                ui->label_selector->setText(QString::fromStdString(str));
            }
        }
        ui->spnFrameRate->setEnabled(true);
        ui->spnExposureTime->setEnabled(true);

        GPUCameraBase::cmrParameterInfo info(GPUCameraBase::prmExposureTime);

        mCameraPtr->getParameterInfo(info);
        {
            QSignalBlocker b(ui->spnExposureTime);
            // ui->spnExposureTime->setMaximum(info.max);
            // ui->spnExposureTime->setMinimum(info.min);
            // ui->spnExposureTime->setSingleStep(info.increment);
            float val = 0;
            if (mCameraPtr->getParameter(GPUCameraBase::prmExposureTime, val))
            {
                ui->spnExposureTime->setValue((int)val);
            }
        }

        {
            QSignalBlocker b1(ui->spnFrameRate);
            // info.param = GPUCameraBase::prmFrameRate;
            // mCameraPtr->getParameterInfo(info);
            // ui->spnFrameRate->setMaximum(info.max);
            // ui->spnFrameRate->setMinimum(info.min);
            // ui->spnFrameRate->setSingleStep(info.increment);
            float val = 0;
            if (mCameraPtr->getParameter(GPUCameraBase::prmFrameRate, val))
            {
                ui->spnFrameRate->setValue((double)val);
            }
        }
    }
}

void CameraSetupWidget::setExposureCamera(float value)
{
    ui->spnExposureTime->setValue(value);
}

void CameraSetupWidget::setFpsCamera(float value)
{
    ui->spnFrameRate->setValue(value);
}

void CameraSetupWidget::on_spnFrameRate_valueChanged(double arg1)
{
    if (mCameraPtr == nullptr)
        return;
    if (mCameraPtr->state() == GPUCameraBase::cstClosed)
        return;

    mCameraPtr->setParameter(GPUCameraBase::prmFrameRate, arg1);

    QSignalBlocker b(ui->spnExposureTime);
    float val = 0;
    if (mCameraPtr->getParameter(GPUCameraBase::prmExposureTime, val))
    {
        ui->spnExposureTime->setValue((int)val);
    }
}

void CameraSetupWidget::on_spnExposureTime_valueChanged(int arg1)
{
    if (mCameraPtr == nullptr)
        return;

    if (mCameraPtr->state() == GPUCameraBase::cstClosed)
        return;

    mCameraPtr->setParameter(GPUCameraBase::prmExposureTime, arg1);

    QSignalBlocker b1(ui->spnFrameRate);
    float val = 0;
    if (mCameraPtr->getParameter(GPUCameraBase::prmFrameRate, val))
    {
        ui->spnFrameRate->setValue((double)val);
    }
}


void CameraSetupWidget::on_radioButton_int_clicked(bool checked)
{
    if (mCameraPtr == nullptr)
        return;
    if (mCameraPtr->devID() < 0)
        return;
    bool ret = mCameraPtr->setParameter(GPUCameraBase::prmTriggerMode, 0);
    if (ret == false)
    {
        QMessageBox::warning(this, "Error", "Failed to set trigger mode");
    }
    else
    {
        ui->lineEdit_frames->setText("0");
        mCameraPtr->setParameter(GPUCameraBase::prmAcqFrameCount, 0);
        emit modeChanged(INPUT_MODE::MODE_CAMERA_INTERNAL_TRIGGER);
    }
}

void CameraSetupWidget::on_radioButton_ext_clicked(bool checked)
{
    if (mCameraPtr == nullptr)
        return;
    if (mCameraPtr->devID() < 0)
        return;
    bool ret = mCameraPtr->setParameter(GPUCameraBase::prmTriggerMode, 1);
    if (ret == false)
    {
        QMessageBox::warning(this, "Error", "Failed to set trigger mode");
    }
    else
    {
        std::ifstream ifs("config.txt");
        if (ifs)
        {
            std::string str;
            ifs >> str;
            ifs.close();
            ui->lineEdit_frames->setText(QString::fromStdString(str));
        }
        mCameraPtr->setParameter(GPUCameraBase::prmAcqFrameCount, ui->lineEdit_frames->text().toInt());

        emit modeChanged(INPUT_MODE::MODE_CAMERA_EXTERNAL_TRIGGER);
    }
}

void CameraSetupWidget::on_pushButton_toFile_clicked()
{
    if (mCameraPtr == nullptr)
        return;
    if (mCameraPtr->devID() < 0)
        return;

    // 打开保存文件对话框
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"), "features.txt",
                                                    tr("Text Files (*.txt)"));
    if (fileName.isEmpty())
    {
        qDebug("User cancelled the dialog or no file was selected.");
    }
    else
    {
        qDebug("Save to file: %s\n", fileName.toStdString().c_str());
        // 在这里处理用户选择的文件路径，例如保存文件
        int ret = mCameraPtr->WriteStreamables(fileName.toStdString());
        if (ret < 0)
        {
            // 弹出错误对话框
            QMessageBox::critical(this, tr("Error"), tr("Failed to save the file. Error code: %1").arg(ret));
        }
    }
}

void CameraSetupWidget::on_pushButton_toDev_clicked()
{
    if (mCameraPtr == nullptr)
        return;
    if (mCameraPtr->devID() < 0)
        return;

    // 打开保存文件对话框
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), "features.txt",
                                                    tr("Text Files (*.txt)"));
    qDebug("load to device from: %s\n", fileName.toStdString().c_str());
    int ret = mCameraPtr->ReadStreamables(fileName.toStdString());
    if (ret < 0)
    {
        // 弹出错误对话框
        QMessageBox::critical(this, tr("Error"), tr("Failed to load the file. Error code: %1").arg(ret));
    }
    setCamera(mCameraPtr);
}

void CameraSetupWidget::on_lineEdit_frames_editingFinished()
{
    if (mCameraPtr)
    {
        if (ui->radioButton_ext->isChecked())
        {
            if (ui->lineEdit_frames->text().toInt() == 0)
            {
                QMessageBox::warning(this, "Error", "Frame count must be greater than 0");
                ui->lineEdit_frames->setText("1");
                mCameraPtr->setParameter(GPUCameraBase::prmAcqFrameCount, 1);
                return;
            }
            mCameraPtr->setParameter(GPUCameraBase::prmAcqFrameCount, ui->lineEdit_frames->text().toInt());
            std::ofstream ofs("config.txt");
            if (!ofs)
            {
                qDebug("open file fail");
            }
            ofs << ui->lineEdit_frames->text().toStdString();
            ofs.close();
        }
        else
        {
            mCameraPtr->setParameter(GPUCameraBase::prmAcqFrameCount, ui->lineEdit_frames->text().toInt());
        }
    }
}
