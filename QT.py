# -*- coding: utf-8 -*-

import sys
from cv2 import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from pyrealsense2 import pyrealsense2 as rs
import os
import time
import png
import threading
import queue
import numpy as np
import math

from yolo6D.Predict import predict, predict_thread, draw_predict
from camera import Camera
import corner
import circle_fit
from yolo6D.darknet import Darknet as dn

RECORD_LENGTH = 18


def make_directories(folder):
    if not os.path.exists(folder+"JPEGImages/"):
        os.makedirs(folder+"JPEGImages/")
    if not os.path.exists(folder+"depth/"):
        os.makedirs(folder+"depth/")


def index2(list_, name):
    '''在result中查找对应物品名称，并返回所在行的索引，未找到则返回-1'''
    for i in range(len(list_)):
        if list_[i][0] == name:
            return i
    return -1  # 未找到


def ReadData():
    '''读取桌面图像处理参数'''
    fp = open('data1.txt', 'r')
    lines = fp.readlines()
    options = dict()
    x = np.zeros((6, 1))
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    x[0] = options['red_low']
    x[1] = options['red_high']
    x[2] = options['green_low']
    x[3] = options['green_high']
    x[4] = options['blue_low']
    x[5] = options['blue_high']
    fp.close()

    ca = np.zeros((3, 1))
    ho = np.zeros((3, 1))
    fp = open('data2.txt', 'r')
    lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    ca[0] = options['canny_threshold_1']
    ca[1] = options['canny_threshold_2']
    ca[2] = options['sobel']
    ho[0] = options['hough_rho']
    ho[1] = int(options['hough_theta']) * math.pi/180
    ho[2] = options['hough_threshold']
    fp.close()
    return x, ca, ho


class Ui_MainWindow(object):
    count = -1  # 玄学照片计数器
    isSquare = True  # 是否是方桌，由下拉菜单控制
    round = 1  # 回合计数器，用于保存结果
    names = []  # 要识别的物品名称（代号）
    models = []  # 预加载模型的存储
    result = []  # 单回合的数据存储

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 700)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.comboBox = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox.setGeometry(QtCore.QRect(1075, 550, 93, 28))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setItemText(3, "")
        self.START = QtWidgets.QPushButton(self.centralWidget)
        self.START.setGeometry(QtCore.QRect(950, 550, 93, 28))
        self.START.setObjectName("START")
        self.STOP = QtWidgets.QPushButton(self.centralWidget)
        self.STOP.setGeometry(QtCore.QRect(1200, 550, 93, 28))
        self.STOP.setObjectName("STOP")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(270, 40, 72, 15))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_2.setGeometry(QtCore.QRect(850, 60, 91, 20))
        self.label_2.setObjectName("label_2")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralWidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(50, 89, 640, 480))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(
            self.horizontalLayoutWidget)
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
        self.type = QtWidgets.QToolButton()
        self.type.setCheckable(True)
        self.timer_camera = QtCore.QTimer()
        self.x, self.ca, self.ho = ReadData()
        print("\033[0;34m开始初始化相机\033[0m")
        camera = Camera()
        camera.init()
        print("\033[0;32m相机初始化完成\033[0m")
        self.thread_init()
        print("\033[0;32m多线程初始化完成\033[0m")
        self.names.append("ZA001")  # 舒肤佳
        # self.names.append("ZA001v")
        self.names.append("ZA001h1")
        # self.names.append("ZA001h2")
        # self.names.append("ZA002")  # 洗手液
        # self.names.append("ZA003")  # 牙膏
        self.names.append("ZA004")  # 花露水
        # self.names.append("ZA005")  # 鸭子
        self.names.append("ZB001")  # 八宝粥
        # self.names.append("ZB002")  # 辣酱
        # self.names.append("ZB003")  # 曲奇
        # self.names.append("ZB004")  # 果珍
        # self.names.append("ZB005")  # 绿箭
        # self.names.append("ZB006")  # 面
        self.names.append("ZB006r")
        self.names.append("ZB007")  # 太平饼干
        self.names.append("ZB007r")
        # self.names.append("ZB008")  # 可比克
        # self.names.append("ZB009")  # 乐事
        # self.names.append("ZB009r")
        # self.names.append("ZB010")  # 瓜子
        self.names.append("ZC004")  # 红牛
        # self.names.append("ZC005")  # AD奶
        # self.names.append("ZC006")  # 橙汁
        # self.names.append("ZC008")  # 加多宝
        # self.names.append("ZC009")  # 冰红茶
        # self.names.append("ZC0091")
        # self.names.append("ZC010")  # 绿茶
        # self.names.append("ZC011")  # 冰糖雪梨
        # self.names.append("ZC011h")
        # self.names.append("ZC012")  # 茶π
        # self.names.append("ZC012h")
        self.names.append("ZC013")  # 椰奶
        self.names.append("ZC014")  # 农夫山泉
        for name in self.names:
            self.result.append([name, 0, 0, 0, 0, 0])
            model = dn('yolo6D/yolo-pose.cfg')
            model.load_weights('weights/' + name + '.weights')
            self.models.append(model)
            print("\033[0;32m%s网络加载完成\033[0m" % name)
        self.START.clicked.connect(lambda: self.open_camera(camera))
        self.timer_camera.timeout.connect(lambda: self.capture_camera(camera))
        self.STOP.clicked.connect(lambda: self.close_camera(camera))
        self.STOP.clicked.connect(self.save_result)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.START.setText(_translate("MainWindow", "开始识别"))
        self.STOP.setText(_translate("MainWindow", "结束识别"))
        self.type.setText(_translate("MainWindow", "type"))
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
        self.comboBox.setItemText(0, _translate("MainWindow", "静态"))
        self.comboBox.setItemText(1, _translate("MainWindow", "动态"))

    def display(self, datas):
        '''
        输入：data：物品数*5数组
        '''
        _translate = QtCore.QCoreApplication.translate
        self.num.setText(_translate("MainWindow", str(len(self.names))))
        if len(datas) >= 1:
            self.X1.setText(_translate("MainWindow", str(datas[0][1])))
            self.Y1.setText(_translate("MainWindow", str(datas[0][2])))
            self.R1.setText(_translate("MainWindow", str(datas[0][3])))
            self.A1.setText(_translate("MainWindow", str(datas[0][4])))
        if len(datas) >= 2:
            self.X2.setText(_translate("MainWindow", str(datas[1][1])))
            self.Y2.setText(_translate("MainWindow", str(datas[1][2])))
            self.R2.setText(_translate("MainWindow", str(datas[1][3])))
            self.A2.setText(_translate("MainWindow", str(datas[1][4])))
        if len(datas) >= 3:
            self.X3.setText(_translate("MainWindow", str(datas[2][1])))
            self.Y3.setText(_translate("MainWindow", str(datas[2][2])))
            self.R3.setText(_translate("MainWindow", str(datas[2][3])))
            self.A3.setText(_translate("MainWindow", str(datas[2][4])))
        if len(datas) >= 4:
            self.X4.setText(_translate("MainWindow", str(datas[3][1])))
            self.Y4.setText(_translate("MainWindow", str(datas[3][2])))
            self.R4.setText(_translate("MainWindow", str(datas[3][3])))
            self.A4.setText(_translate("MainWindow", str(datas[3][4])))

    def save_result(self):
        f = open('东南大学-LabVIEW-R%s.txt' % self.round, 'w+')
        f.write('START\n')
        i = 0
        while(i < len(self.result)):
            if len(self.result[i][0]) > 5:  # 非一对一数据集
                origin = index2(self.result, self.result[i][0][0:5])  # 找到原数据集索引
                if origin == -1:  # 未找到，这种情况不应该出现才对
                    self.result[i][0] = self.result[i][0][0:5]  # 修改为正常输出
                else:
                    if self.result[origin][1] < self.result[i][1]:  # 原数据集置信度不够高
                        self.result[origin] = self.result[i]  # 直接覆盖
                        self.result[origin][0] = self.result[origin][0][0:5]
                    del self.result[i]  # 删除该结果
                    i -= 1
            i += 1
        if self.isSquare:
            for res in self.result:
                if res[1] == 0:  # 无有效识别
                    continue
                f.write('Goal_ID=%s;' % res[0])
                f.write('Goal_X=%.1f;' % (res[2] / res[1]))
                f.write('Goal_Y=%.1f;' % (res[3] / res[1]))
                f.write('Goal_Angle=%.1f\n' % (res[5] / res[1]))
                res[1] = res[2] = res[3] = res[4] = res[5] = 0
        else:
            for res in self.result:
                if res[1] == 0:  # 无有效识别
                    continue
                f.write('Goal_ID=%s;' % res[0])
                f.write('Goal_Radius=%.1f\n' % (res[4] / res[1]))
                res[1] = res[2] = res[3] = res[4] = res[5] = 0
        f.write('END')
        print('    \33[0;32m第%s回合结果已保存\033[0m' % self.round)
        self.round += 1
        f.close()

    def open_camera(self, camera):
        if self.timer_camera.isActive() == False:
            self.timer_camera.start(500)
        # camera = Camera()

    def thread_init(self):
        self.q = queue.Queue(maxsize=len(self.names))  # 状态队列
        # self.q = []
        self.numq = queue.Queue(maxsize=len(self.names))  # 照片编号队列
        self.strs = queue.Queue(maxsize=len(self.names))  # 物品名称和置信度队列

    def show(self, img):
        '''img为图片数据'''
        show = cv2.resize(img, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(
            show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_3.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def capture_camera(self, camera):
        '''拍照'''
        self.isSquare = (self.comboBox.currentText() == "静态")
        self.count = self.count + 1
        print("    \033[0;34m拍摄图片%s.jpg...\033[0m" % self.count)
        d, c = camera.capture()

        filecad = folder+"JPEGImages/%s.jpg" % self.count
        # filedepth = folder+"depth/%s.png" % self.count
        cv2.imwrite(filecad, c)
        self.show(c)
        # cv2.imwrite(filedepth, d)
        # with open(filedepth, 'wb') as f:
        #     writer = png.Writer(
        #         width=d.shape[1], height=d.shape[0], bitdepth=16, greyscale=True)
        #     zgray2list = d.tolist()
        #     writer.write(f, zgray2list)
        print("    \033[0;32m%s.jpg已拍摄\033[0m" % self.count)

        print("    \033[0;34m定位图片%s.jpg...\033[0m" % self.count)
        if self.isSquare:
            lined, Table_2D = corner.square_desk(
                self.count, self.x, self.ca, self.ho)
        else:
            lined, Circle_2D = circle_fit.circle_desk(
                self.count, self.x, self.ca, self.ho)
        print("    \033[0;32mmarked%s.jpg已保存\033[0m" % self.count)

        # 预测
        print("    \033[0;34m预测图片%s.jpg...\033[0m" % self.count)
        threads = []
        for name, model in zip(self.names, self.models):
            threads.append(predict_thread(
                self.q, name, model, self.numq, self.strs))
        starttime = time.time()
        for th in threads:
            self.numq.put(self.count)
            th.start()

        num_done = 0
        bss = []  # 存储物品九点列表
        ret = []  # 存储物品名称和置信度
        datas = []  # 存储结果：[名称，置信度]，x，y，r，a

        while True:
            bss.append(self.q.get())
            ret.append(self.strs.get())
            datas.append([ret[num_done]])  # 名称，置信度
            num_done += 1
            if num_done is len(self.names):
                break

        for bs, data in zip(bss, datas):
            # 根据不同的物品模型，设定不同的底部四点（看预测输出照片可知）
            # lowesty = 0  # 最低
            # for point in bs:

            if data[0][0] == 'ZA001':
                points = [bs[3], bs[4], bs[7], bs[8]]
            elif data[0][0] == 'ZA004' or data[0][0] == 'ZB005' or data[0][0] == 'ZB001' or data[0][0] == 'ZC010' or data[0][0] == 'ZC013' or data[0][0] == 'ZC014':
                points = [bs[2], bs[4], bs[6], bs[8]]
            elif data[0][0] == 'ZA002' or data[0][0] == 'ZB002' or data[0][0] == 'ZB008' or data[0][0] == 'ZB009' or data[0][0] == 'ZC005' or data[0][0] == 'ZC006' or data[0][0] == 'ZC012':
                points = [bs[1], bs[3], bs[5], bs[7]]
            elif data[0][0] == 'ZB003' or data[0][0] == 'ZA001h2':
                points = [bs[1], bs[2], bs[3], bs[4]]
            elif data[0][0] == 'ZB007':
                points = [bs[5], bs[6], bs[7], bs[8]]
            else:
                points = [bs[1], bs[2], bs[3], bs[4]]
            # 将物品坐标（0~1）转为像素坐标
            points = np.matmul(points, [[640, 0], [0, 480]])
            points = np.append(
                points, [[1], [1], [1], [1]], axis=1)  # 将4*2矩阵补为4*3矩阵
            if self.isSquare:
                transed, angle = corner.square_trans(Table_2D, points)
                avgx = np.mean([transed[i][0] for i in range(4)])
                avgy = np.mean([transed[i][1] for i in range(4)])
                data.append('%.1f' % (avgx / 10))  # x
                data.append('%.1f' % (avgy / 10))  # y
                data.append('')  # radius
                data.append('%.1f' % angle)  # angle
            else:
                transed = circle_fit.circle_trans(Circle_2D, points, lined)
                avgx = np.mean([transed[i][0] for i in range(4)])
                avgy = np.mean([transed[i][1] for i in range(4)])
                radius = ((avgx - 300) ** 2 + (avgy - 300) ** 2) ** 0.5
                data.append('')  # x
                data.append('')  # y
                data.append('%.1f' % (radius / 10))  # radius
                data.append('')  # angle
            del points

        print("    \033[0;34m绘制预测中...\033[0m")
        draw_predict(bss, ret, lined, self.count)
        print("        \033[0;34m用时%s秒\033[0m" % (time.time() - starttime))
        print("    \033[0;32m%s.jpg已保存\033[0m" % self.count)
        predicted = cv2.imread('JPEGImages/predict%s.jpg' % self.count, 1)

        self.show(predicted)
        self.display(datas)
        for data in datas:
            for res in self.result:
                if data[0][0] == res[0]:  # 名称一致
                    if self.isSquare:
                        if (float(data[1]) >= 55 or float(data[1]) <= 0 or float(data[2]) >= 55 or float(data[2]) <= 0):  # 错误识别
                            continue
                        trust = float(data[0][1].strip('%'))  # 置信度
                        if trust < 10:  # 低劣识别
                            continue
                        res[1] += trust / 100  # 置信度之和
                        res[2] += float(data[1]) * trust / 100  # x之和
                        res[3] += float(data[2]) * trust / 100  # y之和
                        res[5] += float(data[4]) * trust / 100  # a之和
                    else:
                        if float(data[3]) >= 30:  # 错误识别
                            continue
                        trust = float(data[0][1].strip('%'))  # 置信度
                        if trust < 10:  # 低劣识别
                            continue
                        res[1] += trust / 100  # 置信度之和
                        res[4] += float(data[3]) * trust / 100  # r之和

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
