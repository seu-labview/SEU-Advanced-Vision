from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import pyrealsense2 as rs
import corner
import os
import sys
import copy
from skimage import morphology

import corner
from yolo6D.utils import get_camera_intrinsic
from Kmeans import Cluster


def imageRotate(image, angle, central):
    #图像旋转操作, 同时返回图像中心旋转后的坐标
    row,col=image.shape[:2]
    # print(image.shape)
    #getRotationMatrix2D()，这个函数需要三个参数，
    ang = angle - 90
    #旋转中心，旋转角度（逆 时针），旋转后图像的缩放比例，比如下例为1：
    M=cv2.getRotationMatrix2D((central[0], central[1]), ang, 1)
    #第一个参数为输入图像，第二个参数为仿射变换矩阵，第三个参数为变换后大小,第四个参数为边界外填充颜色
    dst=cv2.warpAffine(image,M,(row,col),borderValue=(255,255,255))
    central = np.array(central)
    central = np.r_[central, [1]]
    trans_c = np.matmul(np.array(M), central.T)
    ret = tuple(trans_c)
    ret1 = int(ret[0]) 
    ret2 = int(ret[1])
    return dst, (ret1, ret2)
    #cv.namedWindow("src",0)
    #cv.imshow("src",image)
    #cv.namedWindow("dst",0)
    #cv.imshow("dst",dst)


def imageresize(image, rate:', central'):
    # img = cv2.imread('flower.jpg')
    # 插值：interpolation
    # None本应该是放图像大小的位置的，后面设置了缩放比例，
    #所有就不要了
    res1 = cv2.resize(image,None,fx=1,fy = rate,interpolation=cv2.INTER_CUBIC)
    # res3 = cv2.resize(image, None, dst=None, fx=None, fy=None, interpolation=None) 
    #直接规定缩放大小，这个时候就不需要缩放因子
    height,width = image.shape[:2]
    res2 = cv2.resize(image,(width,1*height),interpolation=cv2.INTER_CUBIC)
    # trans_c = (int(central[0]), int(rate * central[1]))
    return res1 # , trans_c

def drawpoi(image, point):
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8

    # 要画的点的坐标

    # for point in points_list:
    cv2.circle(image, point, point_size, point_color, thickness)


def circle_canny(img, canny):

    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # threshold
    # lower_gray = np.array([140])
    # upper_gray = np.array([200])
    # test_gray = cv2.inRange(gray,lower_gray,upper_gray)
    # imgray = cv2.Canny(img, 300, 120, 3)
    edges = cv2.Canny(img, canny[0], canny[1], canny[2])
    return edges


def circle_line(origin, thresed, S_thre=(100000, 600000)):
    # imgray = cv2.Canny(img, 300, 120, 3)
    # ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thres", thresh)
    # image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(thresed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    top = []; bottle = []; right = []; left = []; center = []
    Circle_2D = []
    last_longest = 4
    des_ell = tuple()
    for cnt in contours:
        if len(cnt) > last_longest:  # 确保输出点数最多（且面积不超标）的椭圆
            ell=cv2.fitEllipse(cnt) # 返回椭圆所在的矩形框# 函数返回值为：椭圆的中心坐标，长短轴长度（2a，2b），旋转角度
            S2 =math.pi*ell[1][0]*ell[1][1] # 椭圆面积
            if S2>= S_thre[0] and S2 <= S_thre[1]: # 过滤面积小的
                last_longest = len(cnt)
                des_ell = ell

                top_x = ell[0][0] - 0.5 * ell[1][0] * math.cos(math.radians(ell[2]))
                top_y = ell[0][1] - 0.5 * ell[1][0] * math.sin(math.radians(ell[2]))
                bottle_x = ell[0][0] + 0.5 * ell[1][0] * math.cos(math.radians(ell[2]))
                bottle_y = ell[0][1] + 0.5 * ell[1][0] * math.sin(math.radians(ell[2]))

                right_x = ell[0][0] + 0.5 * ell[1][1] * math.sin(math.radians(ell[2]))
                right_y = ell[0][1] - 0.5 * ell[1][1] * math.cos(math.radians(ell[2]))
                left_x = ell[0][0] - 0.5 * ell[1][1] * math.sin(math.radians(ell[2]))
                left_y = ell[0][1] + 0.5 * ell[1][1] * math.cos(math.radians(ell[2]))
            
                top = (int(top_x), int(top_y))
                bottle = (int(bottle_x), int(bottle_y))
                right = (int(right_x), int(right_y))
                left = (int(left_x), int(left_y))
                center = (int(ell[0][0]), int(ell[0][1]))

    origin = cv2.ellipse(origin, des_ell, (0, 255, 0), 2) # 绘制椭圆

    drawpoi(origin, top)
    drawpoi(origin, bottle)
    drawpoi(origin, right)
    drawpoi(origin, left)
    drawpoi(origin, center)
            
            # rate = ell[1][1]/ell[1][0]
            # elif S2< S_thre[1]:
            #     print("\033[0;31m椭圆面积过小！\033[0m")
            # elif S2 > S_thre[0]:
            #     print("\033[0;31m椭圆面积过大！\033[0m")
        # else:
        #     print("\033[0;31m未检测到圆桌！\033[0m")
        #     top = [0, 0]; bottle = [0, 0]; right = [0, 0]; left = [0, 0]; center = [0, 0]

    Circle_2D.append(top); Circle_2D.append(bottle); Circle_2D.append(right); Circle_2D.append(left)
    Circle_2D = np.array(Circle_2D, dtype='float32')
    return origin, thresed, Circle_2D

def circle_desk(num, x, canny, hough):
    '''
    输入：图片编号
    输出：坐标系
    导出：带坐标系图片
    '''
    img_path = 'JPEGImages/' + str(num) + '.jpg'
    img = cv2.imread(img_path, 1)

    thresed = corner.thres(img, x)
    thresed_new = corner.remove_small_objects(thresed)
    lined, _, Circle_2D = circle_line(img, thresed_new, [100000, 600000])
    # affine_table_2D = np.float32([[0,0],[0,550],[550,0],[550,550]])
    # M = cv2.getPerspectiveTransform(Table_2D,affine_table_2D)
    # marked = cv2.warpPerspective(lined,M,(550,550)) # Perspective_Transformation
    cv2.imwrite('JPEGImages/marked' + str(num) + '.jpg', lined)
    return lined, Circle_2D


def circle_trans(Cicle_2D: '上下右左', points: '物体底部四点', lined_img=[]):
    '''
    若最后一个参数非空（调试模式），则显示图片
    '''
    affined_Circle_2D = np.float32(
        [[300, 0], [300, 600], [600, 300], [0, 300]])  # 圆桌R = 300
    M = cv2.getPerspectiveTransform(Cicle_2D, affined_Circle_2D)  # 获取透视变换矩阵

    transed_points = np.matmul(points, np.transpose(M))
    for i in range(4):
        transed_points[i][0] = transed_points[i][0] / transed_points[i][2]
        transed_points[i][1] = transed_points[i][1] / transed_points[i][2]

    a = [0 for i in range(4)]
    for i in range(4):
        a[i] = (int(transed_points[i][0]), int(transed_points[i][1]))

    if len(lined_img):
        # Perspective_Transformation
        transed = cv2.warpPerspective(lined_img, M, (600, 600))
        cv2.line(transed, a[0], a[1], (0, 255, 0), 1)
        cv2.line(transed, a[0], a[2], (0, 255, 0), 1)
        cv2.line(transed, a[1], a[3], (0, 255, 0), 1)
        cv2.line(transed, a[2], a[3], (0, 255, 0), 1)
        cv2.imshow('perspective', transed)

    return transed_points

    