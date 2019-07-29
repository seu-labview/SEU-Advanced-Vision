# -*- coding: utf-8 -*-

import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import pyrealsense2 as rs
import numpy as np
import os
import json
import time
import png

RECORD_LENGTH = 18

def make_directories(folder):
    if not os.path.exists(folder+"JPEGImages/"):
        os.makedirs(folder+"JPEGImages/")
    if not os.path.exists(folder+"depth/"):
        os.makedirs(folder+"depth/")

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 700)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(950, 550, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1200, 550, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3=QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(950, 600, 93, 28))
        self.pushButton.setObjectName("pushButton_3")
        self.pushButton_4=QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(1200, 600, 93, 28))
        self.pushButton.setObjectName("pushButton_4")
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
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.pushButton.clicked.connect(self.open_camera1)
        self.pushButton.clicked.connect(self.input1)
        self.pushButton_3.clicked.connect(self.open_camera2)
        self.pushButton_3.clicked.connect(self.input1)
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.open_camera1)
        self.pushButton_2.clicked.connect(self.close_camera1)
        self.pushButton_2.clicked.connect(self.input2)
        self.pushButton_4.clicked.connect(self.close_camera2)
        self.pushButton_4.clicked.connect(self.input2)
        global X
        global Y
        global A
        global R
        global ID 
        global num


        


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "RectSTART"))
        self.pushButton_2.setText(_translate("MainWindow", "RectCLOSE"))
        self.pushButton_3.setText(_translate("MainWindow","CirSTART"))
        self.pushButton_4.setText(_translate("MainWindow","CirCLOSE"))
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
    def input1(self):
        X=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]
        Y=[y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12]
        A=[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12]
        ID=["","","","","","","","","","","",""]
        R=["","","","","","","","","","","",""]
        num=n1
    def input2(self):
        X=["","","","","","","","","","","",""]
        Y=["","","","","","","","","","","",""]
        A=["","","","","","","","","","","",""]
        ID=["","","","","","","","","","","",""]
        R=[r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12]
        num=n1
        

    def open_camera1(self):
        if self.timer_camera.isActive() == False:
            # self.capture = cv2.VideoCapture(1)
            self.timer_camera.start(100)
        global firsttime
        if firsttime:
            global pipeline
            pipeline = rs.pipeline()
            global config
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start pipeline
            profile = pipeline.start(config)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            # Color Intrinsics 
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                                'ppx': intr.ppx, 'ppy': intr.ppy,
                                'height': intr.height, 'width': intr.width}

            
            with open(folder + 'intrinsics.json', 'w') as fp:
                json.dump(camera_parameters, fp)

            align_to = rs.stream.color
            global align
            align = rs.align(align_to)
            # T_start = time.time()            
            # FileName = 0
            firsttime = False
        _translate = QtCore.QCoreApplication.translate
        # self.X1.setText(_translate("MainWindow",str(X[0])))
        # self.X2.setText(_translate("MainWindow",str(X[1])))
        # self.X3.setText(_translate("MainWindow",str(X[2])))
        # self.X4.setText(_translate("MainWindow",str(X[3])))
        # self.Y1.setText(_translate("MainWindow",str(Y[0])))
        # self.Y2.setText(_translate("MainWindow",str(Y[1])))
        # self.Y3.setText(_translate("MainWindow",str(Y[2])))
        # self.Y4.setText(_translate("MainWindow",str(Y[3])))
        # self.R1.setText(_translate("MainWindow",str(R[0])))
        # self.R2.setText(_translate("MainWindow",str(R[1])))
        # self.R3.setText(_translate("MainWindow",str(R[2])))
        # self.R4.setText(_translate("MainWindow",str(R[3])))
        # self.A1.setText(_translate("MainWindow",str(A[0])))
        # self.A2.setText(_translate("MainWindow",str(A[1])))
        # self.A3.setText(_translate("MainWindow",str(A[2])))
        # self.A4.setText(_translate("MainWindow",str(A[3])))

    # def show_camera(self):
        # while time.time() -T_start <= RECORD_LENGTH:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        d = np.asanyarray(aligned_depth_frame.get_data())
        c = np.asanyarray(color_frame.get_data())
        
        if True:
            global FileName
            filecad= folder+"JPEGImages/%s.jpg" % FileName
            filedepth= folder+"depth/%s.png" % FileName
            cv2.imwrite(filecad,c)
            with open(filedepth, 'wb') as f:
                writer = png.Writer(width=d.shape[1], height=d.shape[0],
                                    bitdepth=16, greyscale=True)
                zgray2list = d.tolist()
                writer.write(f, zgray2list)

            FileName+=1

        show = cv2.resize(c, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(showImage))
    def open_camera2(self):
        if self.timer_camera.isActive() == False:
            # self.capture = cv2.VideoCapture(1)
            self.timer_camera.start(100)
        global firsttime
        if firsttime:
            global pipeline
            pipeline = rs.pipeline()
            global config
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start pipeline
            profile = pipeline.start(config)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            # Color Intrinsics 
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                                'ppx': intr.ppx, 'ppy': intr.ppy,
                                'height': intr.height, 'width': intr.width}

            
            with open(folder + 'intrinsics.json', 'w') as fp:
                json.dump(camera_parameters, fp)

            align_to = rs.stream.color
            global align
            align = rs.align(align_to)
            # T_start = time.time()            
            # FileName = 0
            firsttime = False
        _translate = QtCore.QCoreApplication.translate
        # self.X1.setText(_translate("MainWindow",str(X[0])))
        # self.X2.setText(_translate("MainWindow",str(X[1])))
        # self.X3.setText(_translate("MainWindow",str(X[2])))
        # self.X4.setText(_translate("MainWindow",str(X[3])))
        # self.Y1.setText(_translate("MainWindow",str(Y[0])))
        # self.Y2.setText(_translate("MainWindow",str(Y[1])))
        # self.Y3.setText(_translate("MainWindow",str(Y[2])))
        # self.Y4.setText(_translate("MainWindow",str(Y[3])))
        # self.R1.setText(_translate("MainWindow",str(R[0])))
        # self.R2.setText(_translate("MainWindow",str(R[1])))
        # self.R3.setText(_translate("MainWindow",str(R[2])))
        # self.R4.setText(_translate("MainWindow",str(R[3])))
        # self.A1.setText(_translate("MainWindow",str(A[0])))
        # self.A2.setText(_translate("MainWindow",str(A[1])))
        # self.A3.setText(_translate("MainWindow",str(A[2])))
        # self.A4.setText(_translate("MainWindow",str(A[3])))

    # def show_camera(self):
        # while time.time() -T_start <= RECORD_LENGTH:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        d = np.asanyarray(aligned_depth_frame.get_data())
        c = np.asanyarray(color_frame.get_data())
        
        if True:
            global FileName
            filecad= folder+"JPEGImages/%s.jpg" % FileName
            filedepth= folder+"depth/%s.png" % FileName
            cv2.imwrite(filecad,c)
            with open(filedepth, 'wb') as f:
                writer = png.Writer(width=d.shape[1], height=d.shape[0],
                                    bitdepth=16, greyscale=True)
                zgray2list = d.tolist()
                writer.write(f, zgray2list)

            FileName+=1

        show = cv2.resize(c, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(showImage))


    def close_camera1(self):
        if self.timer_camera.isActive():
            self.timer_camera.stop()
            # self.capture.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        global FileName
        FileName = 0
        fileName= QtWidgets.QFileDialog.getSaveFileName()
        # f=open(fileName[0],'w')
        # f.write("START\n")
        # for n in range(1,num,1):
        #     f.write("Goal_ID"+ID[n]+";"+"Goal_X"+X[n]+";"+"Goal_Y"+Y[n]+";"+"Goal_Angle"+A[n]+";\n")
        # f.write("END")
        f.close()

    def close_camera2(self):
        if self.timer_camera.isActive():
            self.timer_camera.stop()
            # self.capture.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        global FileName
        FileName = 0
        fileName= QtWidgets.QFileDialog.getSaveFileName()
        f=open(fileName[0],'w')
        f.write("START\n")
        for n in range(1,num,1):
            f.write("Goal_ID"+ID[n]+";"+"Goal_Radius"+R[n]+";\n")
        f.write("END")
        f.close()

if __name__ == "__main__":
    folder = os.getcwd() + "/"
    make_directories(folder)
    global FileName
    FileName = 0
    global firsttime
    firsttime = True
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
