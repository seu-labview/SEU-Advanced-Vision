#include "seumainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    SEUMainWindow w;
    w.show();

    return a.exec();
}
