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

#ifndef GTGWIDGET_H
#define GTGWIDGET_H
#include <QImage>
#include <QLabel>
#include <QWidget>
#include <QPixmap>
#include <QDebug>
// glass to glass test
class GtGWidget : public QWidget
{
    Q_OBJECT
public:
    GtGWidget(QWidget* parent = nullptr);
    void start(){setAnimating(true);}
    void stop(){setAnimating(false);}

    void setImage(unsigned char * ptr, int w, int h, int step) 
    {
        if(w == step)
        {
            QImage img(ptr, w, h, step, QImage::Format_Grayscale8);
            *m_pixmap = QPixmap::fromImage(img);
        }
        else if(step == 3 * w)
        {            
            QImage img(ptr, w, h, step, QImage::Format_RGB888);
            *m_pixmap = QPixmap::fromImage(img);
        }
        else
        {
            // 输出错误信息
            qDebug() << "GtGWidget::setImage: unsupported format";            
        }
        update();
    }

protected:
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;


private:
    void setAnimating(bool enabled);
    bool mAnimating = false;
    qint64 mLastTime = 0;
    qint64 mStartTime = 0;
    int mTimerInterval = 16;

    QPixmap *m_pixmap;
};

#endif // GTGWIDGET_H
