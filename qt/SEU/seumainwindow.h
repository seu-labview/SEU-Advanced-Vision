#ifndef SEUMAINWINDOW_H
#define SEUMAINWINDOW_H

#include <QMainWindow>
#include <seu.h>
#include <QCamera>
#include <QCameraViewfinder>
#include <QCameraImageCapture>
#include <QCameraInfo>
#include <QFile>
#include <QFileDialog>
#include <QDebug>
namespace Ui {
class SEU ;
}
class SEUMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    SEUMainWindow(QWidget *parent = 0);
    ~SEUMainWindow();
    SEU uiSEU;
    SEU *pdis;
private:
    Ui::SEU *ui;


};

#endif // SEUMAINWINDOW_H
