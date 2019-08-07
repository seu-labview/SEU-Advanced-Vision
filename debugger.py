import sys
from PyQt5 import QtCore, QtGui, QtWidgets
# from QtWidgets import QApplication, QMainWindow
import PyQt5
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import corner
import math
import os

global point1, point2

def on_mouse(event, x, y, flags, img):
    global point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 5) 
        cv2.imshow('image', img2)
        min_x = min(point1[0],point2[0])     
        min_y = min(point1[1],point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        cut_img = img[min_y:min_y+height, min_x:min_x+width]
        cv2.imwrite('500.jpg', cut_img)
        #显示RGB三个通道的图像
        
        #方法一：显示在一张图上
        # 按R、G、B三个通道分别计算颜色直方图
        b_hist = cv2.calcHist([cut_img], [0], None, [256], [0, 256])
        g_hist = cv2.calcHist([cut_img], [1], None, [256], [0, 256])
        r_hist = cv2.calcHist([cut_img], [2], None, [256], [0, 256])
        # 显示3个通道的颜色直方图
        plt.plot(b_hist, label='B', color='blue')
        plt.plot(g_hist, label='G', color='green')
        plt.plot(r_hist, label='R', color='red')
        plt.legend(loc='best')
        plt.xlim([0, 256])
        plt.show()
        '''
        #方法二：显示在三张图上
        original_img = cv2.imread("500.jpg")
        img = cv2.resize(original_img,None,fx=0.6,fy=0.6,interpolation = cv2.INTER_CUBIC)
        b, g, r = cv2.split(img)  
  
        histImgB = calcAndDrawHist(b, [255, 0, 0])  
        histImgG = calcAndDrawHist(g, [0, 255, 0])  
        histImgR = calcAndDrawHist(r, [0, 0, 255])  

        cv2.imshow("histImgB", histImgB)  
        cv2.imshow("histImgG", histImgG)  
        cv2.imshow("histImgR", histImgR)  
        plt.show()
        cv2.imshow("Img", img)  
        '''
        '''
        #方法三：
        img = cv2.imread('500.jpg')
        color = ('b','g','r')
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
 
        # 使用Mask计算某区域直方图
        img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        mask = np.zeros(img_gray.shape[:2],np.uint8)
        mask[100:200,100:200] = 255
        masked_img = cv2.bitwise_and(img_gray,img_gray,mask = mask)
        hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
        hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
 
        plt.figure()
        plt.subplot(221)
        plt.imshow(img_gray,'gray')
        plt.subplot(222)
        plt.imshow(mask,'gray')
        plt.subplot(223)
        plt.imshow(masked_img,'gray')
        plt.subplot(224)
        plt.plot(hist_full)
        plt.plot(hist_mask)
        plt.xlim([0,256])
        plt.show()
        '''
#
def calcAndDrawHist(image, color):  
    hist= cv2.calcHist([image], [0], None, [256], [0.0,255.0])  
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)  
    histImg = np.zeros([256,256,3], np.uint8)  
    hpt = int(0.9* 256);  
    
    for h in range(256):  
        intensity = int(hist[h]*hpt/maxVal)  
        cv2.line(histImg,(h,256), (h,256-intensity), color)  
    return histImg


# 二值化
def thres(img, x):
    red_low = x[0]
    red_high = x[1]
    green_low = x[2]
    green_high = x[3]
    blue_low = x[4]
    blue_high = x[5]
    height, width = img.shape[:2]
    im_new = np.zeros((height, width, 1), np.uint8)
    for i in range(height):
        for j in range(width):
            judger = 1
            temp = img[i, j]
            if temp[0] < red_low or temp[0] > red_high:
                judger = 0
            if temp[1] < green_low or temp[1] > green_high:
                judger = 0
            if temp[2] < blue_low or temp[2] > blue_high:
                judger = 0
            if judger:
                im_new[i, j] = 255
    cv2.imshow('binary', im_new)
    return

class Ui_Form(object):
    if len(sys.argv) < 2:
        print('\033[0;31mUsage:\033[0m')
        print('\033[0;31m   python3 debugger.py picture_num_in_`JPEGImages/`\033[0m')
        os._exit(-1)

    num = sys.argv[1]
    img = cv2.imread('JPEGImages/' + str(num) + '.jpg', 1)

    def setupUi(self, Form):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_mouse, self.img)
        Form.setObjectName("Form")
        Form.resize(855, 664)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(230, 30, 89, 23))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(580, 30, 89, 23))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(95, 90, 89, 23))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(77, 180, 89, 23))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(90, 270, 89, 23))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(570, 360, 89, 23))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setGeometry(QtCore.QRect(220, 360, 89, 23))
        self.label_7.setObjectName("label_7")
        self.line = QtWidgets.QFrame(Form)
        self.line.setGeometry(QtCore.QRect(50, 330, 741, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(190, 80, 121, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(Form)
        self.lineEdit_2.setGeometry(QtCore.QRect(190, 170, 121, 41))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(Form)
        self.lineEdit_3.setGeometry(QtCore.QRect(190, 260, 121, 41))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(Form)
        self.lineEdit_4.setGeometry(QtCore.QRect(540, 80, 121, 41))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_5 = QtWidgets.QLineEdit(Form)
        self.lineEdit_5.setGeometry(QtCore.QRect(540, 170, 121, 41))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_6 = QtWidgets.QLineEdit(Form)
        self.lineEdit_6.setGeometry(QtCore.QRect(540, 260, 121, 41))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_7 = QtWidgets.QLineEdit(Form)
        self.lineEdit_7.setGeometry(QtCore.QRect(190, 410, 121, 41))
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.lineEdit_8 = QtWidgets.QLineEdit(Form)
        self.lineEdit_8.setGeometry(QtCore.QRect(190, 500, 121, 41))
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(700, 270, 126, 33))
        self.pushButton.setObjectName("pushButton")
        self.lineEdit_9 = QtWidgets.QLineEdit(Form)
        self.lineEdit_9.setGeometry(QtCore.QRect(190, 590, 121, 41))
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.lineEdit_10 = QtWidgets.QLineEdit(Form)
        self.lineEdit_10.setGeometry(QtCore.QRect(540, 410, 121, 41))
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.lineEdit_11 = QtWidgets.QLineEdit(Form)
        self.lineEdit_11.setGeometry(QtCore.QRect(540, 500, 121, 41))
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.lineEdit_12 = QtWidgets.QLineEdit(Form)
        self.lineEdit_12.setGeometry(QtCore.QRect(540, 590, 121, 41))
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.label_8 = QtWidgets.QLabel(Form)
        self.label_8.setGeometry(QtCore.QRect(90, 420, 51, 23))
        self.label_8.setObjectName("label_8")
        self.label_10 = QtWidgets.QLabel(Form)
        self.label_10.setGeometry(QtCore.QRect(30, 600, 131, 23))
        self.label_10.setObjectName("label_10")
        self.label_9 = QtWidgets.QLabel(Form)
        self.label_9.setGeometry(QtCore.QRect(90, 510, 51, 23))
        self.label_9.setObjectName("label_9")
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setGeometry(QtCore.QRect(460, 598, 51, 23))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(Form)
        self.label_12.setGeometry(QtCore.QRect(430, 420, 71, 23))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(Form)
        self.label_13.setGeometry(QtCore.QRect(430, 510, 71, 23))
        self.label_13.setObjectName("label_13")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(700, 600, 126, 33))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton.clicked.connect(self.setting)
        self.pushButton_2.clicked.connect(self.setting_2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form) 
        cv2.imshow('image', self.img)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Setting", "Setting"))
        self.label.setText(_translate("Setting", "MIN"))
        self.label_2.setText(_translate("Setting", "MAX"))
        self.label_3.setText(_translate("Setting", "Red"))
        self.label_4.setText(_translate("Setting", "Green"))
        self.label_5.setText(_translate("Setting", "Blue"))
        self.label_6.setText(_translate("Setting", "Hough"))
        self.label_7.setText(_translate("Setting", "Canny"))
        self.pushButton.setText(_translate("Setting", "binary"))
        self.label_8.setText(_translate("Setting", "阈值1"))
        self.label_10.setText(_translate("Setting", "Sobel算子大小"))
        self.label_9.setText(_translate("Setting", "阈值2"))
        self.label_11.setText(_translate("Setting", "阈值"))
        self.label_12.setText(_translate("Setting", "ρ的精度"))
        self.label_13.setText(_translate("Setting", "θ的精度"))
        self.pushButton_2.setText(_translate("Setting", "corner"))

    
    def setting(self):
        red_low=self.lineEdit.text()
        red_high=self.lineEdit_4.text()
        green_low=self.lineEdit_2.text()
        green_high=self.lineEdit_5.text()
        blue_low=self.lineEdit_3.text()
        blue_high=self.lineEdit_6.text()
        x = np.zeros((6, 1))
        x[0] = red_low
        x[1] = red_high
        x[2] = green_low
        x[3] = green_high
        x[4] = blue_low
        x[5] = blue_high
        thres(self.img, x)

    def setting_2(self):
        ca = np.zeros((3, 1))
        ho = np.zeros((3, 1))
        # ca = (100, 150, 3)
        # ho = (1, np.pi / 180, 80)
        ca[0] = self.lineEdit_7.text()
        ca[1] = self.lineEdit_8.text()
        ca[2] = self.lineEdit_9.text()
        ho[0] = self.lineEdit_10.text()
        ho1 = self.lineEdit_11.text()
        ho[1] = int(ho1) * math.pi/180
        ho[2] = self.lineEdit_12.text()
        corner.square_desk(self.num ,ca, ho)
        f= open("data.txt","w+")
        f.write(self.lineEdit_3.text()+'\n')
        f.write(self.lineEdit_12.text()+'\n')
        f.write(self.lineEdit_11.text()+'\n')
        f.write(self.lineEdit_2.text()+'\n')
        f.write(self.lineEdit_9.text()+'\n')
        f.write(self.lineEdit_10.text()+'\n')
        f.close()


if __name__ == '__main__':    
    #弹出窗口
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget()
    #改成自己的项目名称
    ui = Ui_Form()
    ui.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())   
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
