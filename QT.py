# -*- coding: utf-8 -*-

import sys
from cv2 import cv2 as cv2
from PyQt5 import QtCore, QtGui, QtWidgets
# import pyrealsense2
from pyrealsense2 import pyrealsense2 as rs
import os
import time
import png
import threading
import queue

from yolo6D.Predict import predict, predict_thread, draw_predict
from camera import Camera
from corner import square_desk

RECORD_LENGTH = 18

def make_directories(folder):
    if not os.path.exists(folder+"JPEGImages/"):
        os.makedirs(folder+"JPEGImages/")
    if not os.path.exists(folder+"depth/"):
        os.makedirs(folder+"depth/")

class Ui_MainWindow(object):
    count = -1
    objectnum = 2
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 700)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.START = QtWidgets.QPushButton(self.centralWidget)
        self.START.setGeometry(QtCore.QRect(950, 550, 93, 28))
        self.START.setObjectName("START")
        self.STOP = QtWidgets.QPushButton(self.centralWidget)
        self.STOP.setGeometry(QtCore.QRect(1200, 550, 93, 28))
        self.STOP.setObjectName("STOP")
        # self.pushButton_3=QtWidgets.QPushButton(self.centralWidget)
        # self.START.setGeometry(QtCore.QRect(950, 600, 93, 28))
        # self.START.setObjectName("pushButton_3")
        # self.pushButton_4=QtWidgets.QPushButton(self.centralWidget)
        # self.START.setGeometry(QtCore.QRect(1200, 600, 93, 28))
        # self.START.setObjectName("pushButton_4")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(270, 40, 72, 15))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_2.setGeometry(QtCore.QRect(850, 60, 91, 20))
        self.label_2.setObjectName("label_2")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralWidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(50, 89, 640, 480))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.num = QtWidgets.QLabel(self.centralWidget)
        self.num.setGeometry(QtCore.QRect(940, 60, 72, 21))
        self.num.setFrameShape(QtWidgets.QFrame.Panel)
        self.num.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.num.setText("")
        self.num.setObjectName("num")
        self.label_4 = QtWidgets.QLabel(self.centralWidget)
        self.label_4.setGeometry(QtCore.QRect(850, 110, 91, 20))
        self.label_4.setObjectName("label_4")
        self.X1 = QtWidgets.QLabel(self.centralWidget)
        self.X1.setGeometry(QtCore.QRect(940, 110, 72, 21))
        self.X1.setFrameShape(QtWidgets.QFrame.Panel)
        self.X1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.X1.setText("")
        self.X1.setObjectName("X1")
        self.label_6 = QtWidgets.QLabel(self.centralWidget)
        self.label_6.setGeometry(QtCore.QRect(1160, 110, 81, 20))
        self.label_6.setObjectName("label_6")
        self.label_8 = QtWidgets.QLabel(self.centralWidget)
        self.label_8.setGeometry(QtCore.QRect(850, 150, 91, 20))
        self.label_8.setObjectName("label_8")
        self.Y1 = QtWidgets.QLabel(self.centralWidget)
        self.Y1.setGeometry(QtCore.QRect(940, 150, 72, 21))
        self.Y1.setFrameShape(QtWidgets.QFrame.Panel)
        self.Y1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.Y1.setText("")
        self.Y1.setObjectName("Y1")
        self.label_10 = QtWidgets.QLabel(self.centralWidget)
        self.label_10.setGeometry(QtCore.QRect(1160, 150, 81, 20))
        self.label_10.setObjectName("label_10")
        self.Y3 = QtWidgets.QLabel(self.centralWidget)
        self.Y3.setGeometry(QtCore.QRect(1250, 150, 72, 21))
        self.Y3.setFrameShape(QtWidgets.QFrame.Panel)
        self.Y3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.Y3.setText("")
        self.Y3.setObjectName("Y3")
        self.label_12 = QtWidgets.QLabel(self.centralWidget)
        self.label_12.setGeometry(QtCore.QRect(800, 190, 131, 20))
        self.label_12.setObjectName("label_12")
        self.A1 = QtWidgets.QLabel(self.centralWidget)
        self.A1.setGeometry(QtCore.QRect(940, 190, 72, 21))
        self.A1.setFrameShape(QtWidgets.QFrame.Panel)
        self.A1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.A1.setText("")
        self.A1.setObjectName("A1")
        self.label_14 = QtWidgets.QLabel(self.centralWidget)
        self.label_14.setGeometry(QtCore.QRect(1110, 190, 131, 20))
        self.label_14.setObjectName("label_14")
        self.A3 = QtWidgets.QLabel(self.centralWidget)
        self.A3.setGeometry(QtCore.QRect(1250, 190, 72, 21))
        self.A3.setFrameShape(QtWidgets.QFrame.Panel)
        self.A3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.A3.setText("")
        self.A3.setObjectName("A3")
        self.label_16 = QtWidgets.QLabel(self.centralWidget)
        self.label_16.setGeometry(QtCore.QRect(850, 20, 121, 16))
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.centralWidget)
        self.label_17.setGeometry(QtCore.QRect(810, 240, 121, 20))
        self.label_17.setObjectName("label_17")
        self.R1 = QtWidgets.QLabel(self.centralWidget)
        self.R1.setGeometry(QtCore.QRect(940, 240, 72, 21))
        self.R1.setFrameShape(QtWidgets.QFrame.Panel)
        self.R1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.R1.setText("")
        self.R1.setObjectName("R1")
        self.label_19 = QtWidgets.QLabel(self.centralWidget)
        self.label_19.setGeometry(QtCore.QRect(1120, 240, 121, 20))
        self.label_19.setObjectName("label_19")
        self.R3 = QtWidgets.QLabel(self.centralWidget)
        self.R3.setGeometry(QtCore.QRect(1250, 240, 72, 21))
        self.R3.setFrameShape(QtWidgets.QFrame.Panel)
        self.R3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.R3.setText("")
        self.R3.setObjectName("R3")
        self.label_21 = QtWidgets.QLabel(self.centralWidget)
        self.label_21.setGeometry(QtCore.QRect(841, 330, 81, 20))
        self.label_21.setObjectName("label_21")
        self.X2 = QtWidgets.QLabel(self.centralWidget)
        self.X2.setGeometry(QtCore.QRect(940, 330, 72, 21))
        self.X2.setFrameShape(QtWidgets.QFrame.Panel)
        self.X2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.X2.setText("")
        self.X2.setObjectName("X2")
        self.label_23 = QtWidgets.QLabel(self.centralWidget)
        self.label_23.setGeometry(QtCore.QRect(841, 370, 81, 20))
        self.label_23.setObjectName("label_23")
        self.Y2 = QtWidgets.QLabel(self.centralWidget)
        self.Y2.setGeometry(QtCore.QRect(940, 370, 72, 21))
        self.Y2.setFrameShape(QtWidgets.QFrame.Panel)
        self.Y2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.Y2.setText("")
        self.Y2.setObjectName("Y2")
        self.label_25 = QtWidgets.QLabel(self.centralWidget)
        self.label_25.setGeometry(QtCore.QRect(791, 410, 131, 20))
        self.label_25.setObjectName("label_25")
        self.A2 = QtWidgets.QLabel(self.centralWidget)
        self.A2.setGeometry(QtCore.QRect(940, 410, 72, 21))
        self.A2.setFrameShape(QtWidgets.QFrame.Panel)
        self.A2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.A2.setText("")
        self.A2.setObjectName("A2")
        self.label_27 = QtWidgets.QLabel(self.centralWidget)
        self.label_27.setGeometry(QtCore.QRect(800, 460, 121, 20))
        self.label_27.setObjectName("label_27")
        self.R2 = QtWidgets.QLabel(self.centralWidget)
        self.R2.setGeometry(QtCore.QRect(940, 460, 72, 21))
        self.R2.setFrameShape(QtWidgets.QFrame.Panel)
        self.R2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.R2.setText("")
        self.R2.setObjectName("R2")
        self.label_29 = QtWidgets.QLabel(self.centralWidget)
        self.label_29.setGeometry(QtCore.QRect(1151, 330, 81, 20))
        self.label_29.setObjectName("label_29")
        self.X4 = QtWidgets.QLabel(self.centralWidget)
        self.X4.setGeometry(QtCore.QRect(1250, 330, 72, 21))
        self.X4.setFrameShape(QtWidgets.QFrame.Panel)
        self.X4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.X4.setText("")
        self.X4.setObjectName("X4")
        self.label_31 = QtWidgets.QLabel(self.centralWidget)
        self.label_31.setGeometry(QtCore.QRect(1151, 370, 81, 20))
        self.label_31.setObjectName("label_31")
        self.Y4 = QtWidgets.QLabel(self.centralWidget)
        self.Y4.setGeometry(QtCore.QRect(1250, 370, 72, 21))
        self.Y4.setFrameShape(QtWidgets.QFrame.Panel)
        self.Y4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.Y4.setText("")
        self.Y4.setObjectName("Y4")
        self.label_33 = QtWidgets.QLabel(self.centralWidget)
        self.label_33.setGeometry(QtCore.QRect(1101, 410, 131, 20))
        self.label_33.setObjectName("label_33")
        self.A4 = QtWidgets.QLabel(self.centralWidget)
        self.A4.setGeometry(QtCore.QRect(1250, 410, 72, 21))
        self.A4.setFrameShape(QtWidgets.QFrame.Panel)
        self.A4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.A4.setText("")
        self.A4.setObjectName("A4")
        self.label_35 = QtWidgets.QLabel(self.centralWidget)
        self.label_35.setGeometry(QtCore.QRect(1111, 460, 121, 20))
        self.label_35.setObjectName("label_35")
        self.R4 = QtWidgets.QLabel(self.centralWidget)
        self.R4.setGeometry(QtCore.QRect(1250, 460, 72, 21))
        self.R4.setFrameShape(QtWidgets.QFrame.Panel)
        self.R4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.R4.setText("")
        self.R4.setObjectName("R4")
        self.X3 = QtWidgets.QLabel(self.centralWidget)
        self.X3.setGeometry(QtCore.QRect(1250, 110, 72, 21))
        self.X3.setFrameShape(QtWidgets.QFrame.Panel)
        self.X3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.X3.setText("")
        self.X3.setObjectName("X3")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1208, 26))
        # self.menuBar.setObjectName("menuBar")
        # MainWindow.setMenuBar(self.menuBar)
        # self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        # self.mainToolBar.setObjectName("mainToolBar")
        # MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        # self.statusBar = QtWidgets.QStatusBar(MainWindow)
        # self.statusBar.setObjectName("statusBar")
        # MainWindow.setStatusBar(self.statusBar)
        self.type = QtWidgets.QToolButton()
        self.type.setCheckable(True)
        round = 1
        self.type.setChecked(round)
        self.type.clicked.connect(lambda:self.changetype(round))
        self.type.setGeometry(QtCore.QRect(1100, 550, 100, 100))
        self.timer_camera = QtCore.QTimer()
        print(" 0% 开始初始化相机")
        camera = Camera()
        camera.init()
        print(" 5% \033[0;32m相机初始化完成\033[0m")
        self.thread_init()
        print("10% \033[0;32m多线程初始化完成\033[0m")
        self.START.clicked.connect(lambda:self.open_camera(camera))
        # self.START.clicked.connect(self.display)
        # self.pushButton_3.clicked.connect(self.open_camera)
        # self.pushButton_3.clicked.connect(self.input1)
        self.timer_camera.timeout.connect(lambda:self.capture_camera(camera))
        self.STOP.clicked.connect(lambda:self.close_camera(camera))
        self.STOP.clicked.connect(self.input2)
        # self.pushButton_4.clicked.connect(lambda:self.close_camera(camera))
        # self.pushButton_4.clicked.connect(self.input2)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.START.setText(_translate("MainWindow", "开始识别"))
        self.STOP.setText(_translate("MainWindow", "结束识别"))
        self.type.setText(_translate("MainWindow", "type"))
        # self.pushButton_3.setText(_translate("MainWindow","CirSTART"))
        # self.pushButton_4.setText(_translate("MainWindow","CirCLOSE"))
        self.label.setText(_translate("MainWindow", "图像显示"))
        self.label_2.setText(_translate("MainWindow", "识别目标数"))
        self.label_4.setText(_translate("MainWindow", "目标1中心X"))
        self.label_6.setText(_translate("MainWindow", "目标3中心X"))
        self.label_8.setText(_translate("MainWindow", "目标1中心Y"))
        self.label_10.setText(_translate("MainWindow", "目标3中心Y"))
        self.label_12.setText(_translate("MainWindow", "目标1朝向角Angle"))
        self.label_14.setText(_translate("MainWindow", "目标3朝向角Angle"))
        self.label_16.setText(_translate("MainWindow", "识别结果输出区"))
        self.label_17.setText(_translate("MainWindow", "目标1距离Radius"))
        self.label_19.setText(_translate("MainWindow", "目标3距离Radius"))
        self.label_21.setText(_translate("MainWindow", "目标2中心X"))
        self.label_23.setText(_translate("MainWindow", "目标2中心Y"))
        self.label_25.setText(_translate("MainWindow", "目标2朝向角Angle"))
        self.label_27.setText(_translate("MainWindow", "目标2距离Radius"))
        self.label_29.setText(_translate("MainWindow", "目标4中心X"))
        self.label_31.setText(_translate("MainWindow", "目标4中心Y"))
        self.label_33.setText(_translate("MainWindow", "目标4朝向角Angle"))
        self.label_35.setText(_translate("MainWindow", "目标4距离Radius"))

    def display(self, data):
        '''
        输入：data：物品数*4数组
        '''
        _translate = QtCore.QCoreApplication.translate
        self.X1.setText(_translate("MainWindow",str(data[[0]][0])))
        self.X2.setText(_translate("MainWindow",str(data[[1]][0])))
        self.X3.setText(_translate("MainWindow",str(data[[2]][0])))
        self.X4.setText(_translate("MainWindow",str(data[[3]][0])))
        self.Y1.setText(_translate("MainWindow",str(data[[0]][1])))
        self.Y2.setText(_translate("MainWindow",str(data[[1]][1])))
        self.Y3.setText(_translate("MainWindow",str(data[[2]][1])))
        self.Y4.setText(_translate("MainWindow",str(data[[3]][1])))
        self.R1.setText(_translate("MainWindow",str(data[[0]][2])))
        self.R2.setText(_translate("MainWindow",str(data[[1]][2])))
        self.R3.setText(_translate("MainWindow",str(data[[2]][2])))
        self.R4.setText(_translate("MainWindow",str(data[[3]][2])))
        self.A1.setText(_translate("MainWindow",str(data[[0]][3])))
        self.A2.setText(_translate("MainWindow",str(data[[1]][3])))
        self.A3.setText(_translate("MainWindow",str(data[[2]][3])))
        self.A4.setText(_translate("MainWindow",str(data[[3]][3])))
        
    def input2(self):
        pass

    def changetype(self, Type):
        Type = ~Type

    def open_camera(self, camera):
        if self.timer_camera.isActive() == False:
            self.timer_camera.start(500)
        # camera = Camera()

    def thread_init(self):
        self.q = queue.Queue(maxsize = self.objectnum) # 状态队列
        # self.q = []
        self.numq = queue.Queue(maxsize = self.objectnum) # 照片编号队列
        self.strs = queue.Queue(maxsize = self.objectnum) # 物品名称和识别率队列

    def show(self, img):
        '''img为图片数据'''        
        show = cv2.resize(img, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def capture_camera(self, camera):
        '''拍照'''
        self.count = self.count + 1
        print("    \033[0;34m拍摄图片%s.jpg...\033[0m" % self.count)
        d, c = camera.capture(self.count)

        filecad= folder+"JPEGImages/%s.jpg" % self.count
        filedepth= folder+"depth/%s.png" % self.count
        cv2.imwrite(filecad,c)
        self.show(c)
        with open(filedepth, 'wb') as f:
            writer = png.Writer(width=d.shape[1], height=d.shape[0], bitdepth=16, greyscale=True)
            zgray2list = d.tolist()
            writer.write(f, zgray2list)

        print("    \033[0;32m%s.jpg已拍摄\033[0m" % self.count)

        # 预测
        print("    \033[0;34m预测图片%s.jpg...\033[0m" % self.count)
        threads = []
        threads.append(predict_thread(self.q, 'safeguard', self.numq, self.strs))
        # threads.append(predict_thread(self.q, 'copico', self.numq))
        threads.append(predict_thread(self.q, 'floral_water', self.numq, self.strs))
        starttime = time.time()
        for th in threads:
            self.numq.put(self.count)
            th.start()

        num_done = 0
        bs = []; ret = []
        
        while True:
            if ~self.q.empty():
                bs.append(self.q.get())
                ret.append(self.strs.get())
                num_done += 1
                if num_done is self.objectnum:
                    break
        
        draw_predict(bs, ret, c, self.count)
        print("        \033[0;34m用时%s秒\033[0m" % (time.time() - starttime))
        print("    \033[0;32m%s.jpg已保存\033[0m" % self.count)
        # predicted = cv2.imread('JPEGImages/%s.jpg' % self.count, 1)
        # self.show(predicted)
        print("    \033[0;34m定位图片%s.jpg...\033[0m" % self.count)
        square_desk(self.count)
        print("    \033[0;32mmarked%s.jpg已保存\033[0m" % self.count)
        # marked_img = 'marked' + str(self.count) + '.jpg'
        marked_img = 'corner.jpg'
        marked = cv2.imread(marked_img,1)
        self.show(marked)

    def close_camera(self, camera):
        if self.timer_camera.isActive():
            self.timer_camera.stop()
            
        del camera
        cv2.destroyAllWindows()
        self.count = 0

if __name__ == "__main__":
    folder = os.getcwd() + "/"
    make_directories(folder)
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
