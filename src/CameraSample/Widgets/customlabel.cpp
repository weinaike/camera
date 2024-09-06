#include "customlabel.h"
#include <QPainter>

CustomLabel::CustomLabel(QWidget *parent)
    : QLabel(parent)
{
}

void CustomLabel::setPixmap(const QPixmap &pixmap)
{
    mPixmap = pixmap;
    update(); // 触发重绘事件
}

void CustomLabel::paintEvent(QPaintEvent *event)
{
    QLabel::paintEvent(event); // 调用基类的 paintEvent 方法

    if (!mPixmap.isNull()) {
        QPainter painter(this);
        painter.drawPixmap(0, 0, mPixmap);
    }
}