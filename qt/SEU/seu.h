#ifndef SEU_H
#define SEU_H

#include <QWidget>
#include <QCamera>
#include <QCameraViewfinder>
#include <QCameraInfo>
#include <QFileDialog>
#include <QDebug>

namespace Ui {
class SEU;
}
class QCamera;
class QCameraViewfinder;

class SEU : public QWidget
{
    Q_OBJECT

public:
    explicit SEU(QWidget *parent = nullptr);
    ~SEU();
private slots:
    void on_pushButton_clicked();


    void on_pushButton_2_clicked();

private:
    Ui::SEU *ui;
    QCamera *camera;
    QCameraViewfinder *viewfinder;
};

#endif // SEU_H
