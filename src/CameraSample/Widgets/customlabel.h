#ifndef CUSTOMLABEL_H
#define CUSTOMLABEL_H

#include <QLabel>
#include <QPixmap>

class CustomLabel : public QLabel
{
    Q_OBJECT

public:
    explicit CustomLabel(QWidget *parent = nullptr);
    void setPixmap(const QPixmap &pixmap);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QPixmap mPixmap;
};

#endif // CUSTOMLABEL_H