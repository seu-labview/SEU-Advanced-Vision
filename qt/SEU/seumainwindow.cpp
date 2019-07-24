#include "seumainwindow.h"
#include "ui_seu.h"
#include <QCamera>
#include <seumainwindow.h>

SEUMainWindow::SEUMainWindow(QWidget *parent)
    : QMainWindow(parent),
    ui(new Ui::SEU)
{
    ui->setupUi(this);
    pdis=new SEU(this);
    setFixedSize(1500,900);
}

SEUMainWindow::~SEUMainWindow()
{
    delete ui;

}
