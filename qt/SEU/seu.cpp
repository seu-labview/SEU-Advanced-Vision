#include "seu.h"
#include "ui_seu.h"
#include <QFile>
#include <QTextStream>
#include <QDataStream>
#include <QDebug>
#include<QCamera>
#include <seumainwindow.h>
#include <QMessageBox>
#include <QFileDevice>
#include <QtEvents>

SEU::SEU(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SEU)
{
    ui->setupUi(this);
    connect(ui->pushButton_2,SIGNAL(clicked()),qApp,SLOT(on_pushButton_2_clicked()));

}

SEU::~SEU()
{
    delete ui;
}
double x1,x2,x3,x4,Y1,Y2,Y3,Y4,r1,r2,r3,r4,a1,a2,a3,a4,num;
void SEU::on_pushButton_clicked()
{
    QString tempStr;
    double x1=3;
    //得到这些参数的值；
    ui->num->setText(tempStr.setNum(num));
    ui->X1->setText(tempStr.setNum(x1));
    ui->X2->setText(tempStr.setNum(x2));
    ui->X3->setText(tempStr.setNum(x3));
    ui->X4->setText(tempStr.setNum(x4));

    ui->Y4->setText(tempStr.setNum(Y4));
    ui->Y3->setText(tempStr.setNum(Y3));
    ui->Y2->setText(tempStr.setNum(Y2));
    ui->Y1->setText(tempStr.setNum(Y1));

    ui->Angel1->setText(tempStr.setNum(a1));
    ui->Angel2->setText(tempStr.setNum(a2));
    ui->Angel3->setText(tempStr.setNum(a3));
    ui->Angel4->setText(tempStr.setNum(a4));

    ui->Radius1->setText(tempStr.setNum(r1));
    ui->Radius2->setText(tempStr.setNum(r2));
    ui->Radius3->setText(tempStr.setNum(r3));
    ui->Radius4->setText(tempStr.setNum(r4));
    camera=new QCamera("@device:pnp:\\\\?\\usb#vid_8086&pid_0aa5&mi_00#7&2df25655&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global",this);
    viewfinder=new QCameraViewfinder(this);
    ui->horizontalLayout->addWidget(viewfinder);
    ui->label_21->setScaledContents(true);
    viewfinder->resize(ui->label_21->size());
    camera->setViewfinder(viewfinder);
    camera->start();

}

void SEU::on_pushButton_2_clicked()//输出文件格式
{
    QFileDialog filedialog;
    QString fileName=QFileDialog::getSaveFileName(this,tr("保存到文件"),QDir::homePath(),tr("txt格式文件(.txt)"));
    QFile file(fileName);
    QTextStream textstream(&file);
    QString str;
    textstream<<str;
    camera->stop();


}
