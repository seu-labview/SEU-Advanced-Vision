import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import pyrealsense2 as rs
import corner
import os
import sys
from yolo6D.utils import get_camera_intrinsic
from Kmeans import Cluster
import copy
from skimage import morphology


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
    # points_list = [(160, 160), (136, 160), (150, 200), (200, 180), (120, 150), (145, 180)]

    # for point in points_list:
    cv2.circle(image, point, point_size, point_color, thickness)

    # 画圆，圆心为：(160, 160)，半径为：60，颜色为：point_color，实心线
    # cv.circle(img, (160, 160), 60, point_color, 0)

    # cv.namedWindow("image")
    # cv.imshow('image', img)
    # cv.waitKey (10000) # 显示 10000 ms 即 10s 后消失
    # cv.destroyAllWindows()


def my_fun(img):
    # img = cv2.imread("test.jpg", 3)
    #img=cv2.blur(img,(1,1))

    # 加入二值化
    """
    threshold_x = np.array([160, 255, 160, 255, 160, 255])
    thresImg = thres(img, threshold_x)
    cv2.imshow("thresImg.jpg", thresImg)
    """

    #原来的算法
    # imgray=cv2.Canny(img, 300, 120, 3)#Canny边缘检测，参数可更改(600, 100), 参数待改

    # x = [170, 255, 170, 255, 170, 255]

    # img1 = corner.thres(img, x)

    imgray = cv2.Canny(img, 300, 120, 3)

    # 加入腐蚀的canny
    # imgray = square_canny(img, (300, 120, 3))

    # cv2.imshow("imgray", imgray)
    ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
    # cv2.imshow("imbinary.jpg", thresh)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#contours为轮廓集，可以计算轮廓的长度、面积等
    for cnt in contours:
        if len(cnt)>270:  # 306
            S1=cv2.contourArea(cnt) # green公式算面积
            ell=cv2.fitEllipse(cnt) # 返回椭圆所在的矩形框# 函数返回值为：椭圆的中心坐标，长短轴长度（2a，2b），旋转角度
            S2 =math.pi*ell[1][0]*ell[1][1] # 椭圆面积
            if S2>= 150000 and S2 <= 600000: # 过滤面积小的
                des_ell = ell
                img = cv2.ellipse(img, des_ell, (0, 255, 0), 2) # 绘制椭圆

                top_x = ell[0][0] - 0.5 * ell[1][0] * math.cos(math.radians(ell[2]))
                top_y = ell[0][1] - 0.5 * ell[1][0] * math.sin(math.radians(ell[2]))
                bottle_x = ell[0][0] + 0.5 * ell[1][0] * math.cos(math.radians(ell[2]))
                bottle_y = ell[0][1] + 0.5 * ell[1][0] * math.sin(math.radians(ell[2]))

                right_x = ell[0][0] + 0.5 * ell[1][1] * math.sin(math.radians(ell[2]))
                right_y = ell[0][1] - 0.5 * ell[1][1] * math.cos(math.radians(ell[2]))
                left_x = ell[0][0] - 0.5 * ell[1][1] * math.sin(math.radians(ell[2]))
                left_y = ell[0][1] + 0.5 * ell[1][1] * math.cos(math.radians(ell[2]))

                top = []; bottle = []; right = []; left = []
                top.append(top_x); top.append(top_y)
                bottle.append(bottle_x); bottle.append(bottle_y)
                right.append(right_x); right.append(right_y)
                left.append(left_x); left.append(left_y)
                
                drawpoi(img, (int(top_x), int(top_y)))
                drawpoi(img, (int(bottle_x), int(bottle_y)))
                drawpoi(img, (int(right_x), int(right_y)))
                drawpoi(img, (int(left_x), int(left_y)))
                drawpoi(img, (int(ell[0][0]), int(ell[0][1])))
                # output, cent_r = imageRotate(output, des_ell[2], (des_ell[1][0], des_ell[1][1])) # 现在桌子转正了
                # drawpoi(output, (int(des_ell[1][0]), int(des_ell[1][1])))

                # drawpoi(output, cent_r)
                # print(cent_r)
                rate = ell[1][1]/ell[1][0]
                # output = imageresize(output, rate) #, cent_s,, cent_r
                # drawpoi(output, cent_s)
                return img, top, bottle, right, left
        else:
            top = [0, 0]; bottle = [0, 0]; right = [0, 0]; left = [0, 0]
    #cv2.imshow("output.jpg",output)
    return img, top, bottle, right, left
    # canvasPoints = np.float([[97., 35.], [505., 35.] ,[23., 368.], [586., 368.]])
    # srcImg = cv2.
    # perspectiveImg = cv2.warpPerspective(srcImg, perspectiveMatrix, (700, 500))
    # cv2.imshow('PerspectiveImg', perspectiveImg)
    # cv2.waitKey(0)


def draw(img, corner, imgpts):
    '''绘制坐标系'''

    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 3)
    cv2.putText(img, "X", tuple(
        imgpts[0].ravel()), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 149, 237), 2)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
    cv2.putText(img, "Y", tuple(imgpts[1].ravel()),
               cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 127), 2)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 3)
    cv2.putText(img, "Z", tuple(imgpts[2].ravel()),
               cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 140, 0), 2)
    return img


def read_data_cfg(datacfg):
    options = dict()
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options



def getcrosspoint(rho1, theta1, rho2, theta2):
    cos_1 = np.cos(theta1)
    sin_1 = np.sin(theta1)
    K1 = -(cos_1 / sin_1)

    cos_2 = np.cos(theta2)
    sin_2 = np.sin(theta2)
    K2 = -(cos_2 / sin_2)

    x_1 = cos_1 * rho1
    y_1 = sin_1 * rho1

    x_2 = cos_2 * rho2
    y_2 = sin_2 * rho2

    b_1 = -(K1 * x_1) + y_1
    b_2 = -(K2 * x_2) + y_2

    corss_x = (b_1 - b_2)/(K2 - K1)
    cross_y = K1 * corss_x + b_1
    corss_x = int(corss_x)
    cross_y = int(cross_y)
    # print(corss_x, cross_y)
    return (corss_x, cross_y)



def thres(img, x):
    '''二值化'''
    red_low = x[0]
    red_high = x[1]
    green_low = x[2]
    green_high = x[3]
    blue_low = x[4]
    blue_high = x[5]
    height, width = img.shape[:2]   #j:高    i:宽
    im_new = np.zeros((height, width, 1), np.uint8)   #粗糙的二值化图像
    for i in range(width):
        for j in range(height):
            judger = 1
            temp = img[j, i]     #j:高    i:宽
            if temp[0] < red_low or temp[0] > red_high:
                judger = 0
            if temp[1] < green_low or temp[1] > green_high:
                judger = 0
            if temp[2] < blue_low or temp[2] > blue_high:
                judger = 0
            if judger:
                im_new[j, i] = 255   #j:高 i:宽
    return im_new


def remove_small_objects(img):
    kernel = np.ones((5, 5), np.uint8)  
    erosion = cv2.erode(img, kernel, iterations = 1)
    # erosion = cv2.dilate(erosion,kernel,iterations = 1)
    erosion = erosion > 127
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    erosion = morphology.remove_small_objects(erosion, min_size=500, connectivity=1)  #0 1
    chull = morphology.convex_hull_image(erosion)
    chull = chull.astype(np.uint8)
    chull *= 255
    return chull


def circle_canny(img, canny):

    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # threshold
    # lower_gray = np.array([140])
    # upper_gray = np.array([200])
    # test_gray = cv2.inRange(gray,lower_gray,upper_gray)
    """
    kernel = np.ones((9, 9), np.uint8)
    test_gray = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    test_gray = cv2.morphologyEx(test_gray, cv2.MORPH_OPEN, kernel)
    kernel2 = np.ones((9, 9), np.uint8)
    test_gray = cv2.morphologyEx(test_gray, cv2.MORPH_CLOSE, kernel2)
    edges = cv2.Canny(test_gray, canny[0], canny[1], canny[2])
    """
    # imgray = cv2.Canny(img, 300, 120, 3)
    edges = cv2.Canny(img, canny[0], canny[1], canny[2])
    return edges


def square_line(origin, edges, hough):
    '''
    输入：原始图片，黑白边线图，hough数组
    输出：带坐标的原图，Table_2D
    '''
    internal_calibration = get_camera_intrinsic()
    internal_calibration = np.array(internal_calibration, dtype='float32')
    distCoeffs = np.zeros((8, 1), dtype='float32')
    img = copy.copy(origin)
    lines = cv2.HoughLines(edges, hough[0], hough[1], hough[2])

    # rho：ρ，图片左上角向直线所做垂线的长度
    # theta：Θ，图片左上角向直线所做垂线与顶边夹角
    # 垂足高于原点时，ρ为负，Θ为垂线关于原点的对称线与顶边的夹角
    top_line_theta = []
    top_line_rho = []
    left_line_theta = []
    left_line_rho = []
    right_line_theta = []
    right_line_rho = []
    bottom_line_theta = []
    bottom_line_rho = []
    horizon = []
    summ = 0
    final_lines = np.zeros((4, 2))
    if len(lines) < 4:
        print("\033[0;31m未检测到方桌！\033[0m")
        return edges
    else:
        for line in lines:
            for rho, theta in line:
                if (theta > math.pi / 3 and theta < math.pi * 2 / 3):  # 横线
                    horizon.append(line)
                elif rho < 0:  # 右边
                    right_line_rho.append(rho)
                    right_line_theta.append(theta)
                else:  # 左边
                    left_line_theta.append(theta)
                    left_line_rho.append(rho)
        top, bottom = Cluster(horizon, 180, 360)  # 将横线依据abs(rho)分为上下
        for line in top:
            for rho, theta in line:
                top_line_rho.append(rho)
                top_line_theta.append(theta)
        for line in bottom:
            for rho, theta in line:
                bottom_line_rho.append(rho)
                bottom_line_theta.append(theta)

        for i in right_line_theta:
            summ += i
        right_line_theta_average = summ / len(right_line_theta)
        final_lines[0, 1] = right_line_theta_average
        summ = 0
        for i in right_line_rho:
            summ += i
        right_line_rho_average = summ / len(right_line_rho)
        final_lines[0, 0] = right_line_rho_average
        summ = 0

        for i in left_line_theta:
            summ += i
        left_line_theta_average = summ / len(left_line_theta)
        final_lines[1, 1] = left_line_theta_average
        summ = 0
        for i in left_line_rho:
            summ += i
        left_line_rho_average = summ / len(left_line_rho)
        final_lines[1, 0] = left_line_rho_average
        summ = 0

        for i in top_line_theta:
            summ += i
        top_line_theta_average = summ / len(top_line_theta)
        final_lines[2, 1] = top_line_theta_average
        summ = 0
        for i in top_line_rho:
            summ += i
        top_line_rho_average = summ / len(top_line_rho)
        final_lines[2, 0] = top_line_rho_average
        summ = 0

        for i in bottom_line_theta:
            summ += i
        bottom_line_theta_average = summ / len(bottom_line_theta)
        final_lines[3, 1] = bottom_line_theta_average
        summ = 0
        for i in bottom_line_rho:
            summ += i
        bottom_line_rho_average = summ / len(bottom_line_rho)
        final_lines[3, 0] = bottom_line_rho_average
        summ = 0
        # print(final_lines)
        final_lines = np.array(final_lines)
        for i in range(4):
            theta = final_lines[i, 1]
            rho = final_lines[i, 0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1, y1), (x2, y2), (200, 135, 100), 2)
            # 标记直线编号
            string = str(i)
            cv2.putText(img, string, (int(x0), int(y0)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)

        left_top_point_x, left_top_point_y = getcrosspoint(
            left_line_rho_average, left_line_theta_average,
            top_line_rho_average, top_line_theta_average)
        left_bottom_point_x, left_bottom_point_y = getcrosspoint(
            left_line_rho_average, left_line_theta_average,
            bottom_line_rho_average, bottom_line_theta_average)
        right_top_point_x, right_top_point_y = getcrosspoint(
            right_line_rho_average, right_line_theta_average,
            top_line_rho_average, top_line_theta_average)
        right_bottom_point_x, right_bottom_point_y = getcrosspoint(
            right_line_rho_average, right_line_theta_average,
            bottom_line_rho_average, bottom_line_theta_average)

        Table_2D = []
        Table_2D.append([left_top_point_x, left_top_point_y])
        Table_2D.append([left_bottom_point_x, left_bottom_point_y])
        Table_2D.append([right_top_point_x, right_top_point_y])
        Table_2D.append([right_bottom_point_x, right_bottom_point_y])
        cv2.circle(img, (left_top_point_x, left_top_point_y),
                   3, (255, 0, 0), -1)
        cv2.circle(img, (left_bottom_point_x, left_bottom_point_y),
                   3, (255, 0, 0), -1)
        cv2.circle(img, (right_top_point_x, right_top_point_y),
                   3, (255, 0, 0), -1)
        cv2.circle(img, (right_bottom_point_x,
                         right_bottom_point_y), 3, (255, 0, 0), -1)
        Table_3D = []
        Table_3D.append([0, 0, 0])
        Table_3D.append([0, 55, 0])
        Table_3D.append([55, 0, 0])
        Table_3D.append([55, 55, 0])
        Table_3D = np.array(Table_3D, dtype='float32')
        Table_2D = np.array(Table_2D, dtype='float32')
        _, rvector, tvector = cv2.solvePnP(
            Table_3D, Table_2D, internal_calibration, distCoeffs)
        axis = np.float32([[55, 0, 0], [0, 55, 0], [0, 0, -20]]).reshape(-1, 3)
        imgpts, _ = cv2.projectPoints(
            axis, rvector, tvector, internal_calibration, distCoeffs,)
        lined = draw(img, (left_top_point_x, left_top_point_y), imgpts)
    return lined, Table_2D


def Circle_line(img, S_thre = (150000, 600000)):
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    top = []; bottle = []; right = []; left = []; center = []
    Circle_2D = []
    for cnt in contours:
        if len(cnt)>270:  # 306
            S1=cv2.contourArea(cnt) 
            ell=cv2.fitEllipse(cnt) # 返回椭圆所在的矩形框# 函数返回值为：椭圆的中心坐标，长短轴长度（2a，2b），旋转角度
            S2 =math.pi*ell[1][0]*ell[1][1] # 椭圆面积
            if S2>= S_thre[1] and S2 <= S_thre[2]: # 过滤面积小的
                des_ell = ell
                img = cv2.ellipse(img, des_ell, (0, 255, 0), 2) # 绘制椭圆

                top_x = ell[0][0] - 0.5 * ell[1][0] * math.cos(math.radians(ell[2]))
                top_y = ell[0][1] - 0.5 * ell[1][0] * math.sin(math.radians(ell[2]))
                bottle_x = ell[0][0] + 0.5 * ell[1][0] * math.cos(math.radians(ell[2]))
                bottle_y = ell[0][1] + 0.5 * ell[1][0] * math.sin(math.radians(ell[2]))

                right_x = ell[0][0] + 0.5 * ell[1][1] * math.sin(math.radians(ell[2]))
                right_y = ell[0][1] - 0.5 * ell[1][1] * math.cos(math.radians(ell[2]))
                left_x = ell[0][0] - 0.5 * ell[1][1] * math.sin(math.radians(ell[2]))
                left_y = ell[0][1] + 0.5 * ell[1][1] * math.cos(math.radians(ell[2]))

                top.append(int(top_x)); top.append(int(top_y))
                bottle.append(int(bottle_x)); bottle.append(int(bottle_y))
                right.append(int(right_x)); right.append(int(right_y))
                left.append(int(left_x)); left.append(int(left_y))
                center.append(int(ell[0][0]), int(ell[0][1]))
                
                drawpoi(img, top)
                drawpoi(img, bottle)
                drawpoi(img, right)
                drawpoi(img, left)
                drawpoi(img, center)
                # output, cent_r = imageRotate(output, des_ell[2], (des_ell[1][0], des_ell[1][1])) # 现在桌子转正了
                # drawpoi(output, (int(des_ell[1][0]), int(des_ell[1][1])))

                # drawpoi(output, cent_r)
                # print(cent_r)
                # rate = ell[1][1]/ell[1][0]
                # output = imageresize(output, rate) #, cent_s,, cent_r
                # drawpoi(output, cent_s)
                # return img, top, bottle, right, left
        else:
            top = [0, 0]; bottle = [0, 0]; right = [0, 0]; left = [0, 0]; center = [0, 0]

    Circle_2D.append(top); Circle_2D.append(bottle); Circle_2D.append(right); Circle_2D.append(left)
    #cv2.imshow("output.jpg",output)
    return img, Circle_2D

def Circle_desk(num, x, canny, hough):
    '''
    输入：图片编号
    输出：坐标系
    导出：带坐标系图片
    '''
    img_path = 'JPEGImages/' + str(num) + '.jpg'
    img = cv2.imread(img_path, 1)

    # 暂时不要二值化-----------------------------------------------------------------------------------------------------
    # thresed = thres(img, x)
    thresed = img
    thresed_new = remove_small_objects(thresed)
    edges = circle_canny(thresed_new, canny) # canny = [300, 120, 3]
    # lined, Table_2D = square_line(img, edges, hough)
    lined, Circle_2D = Circle_line(img, [150000, 600000])
    # affine_table_2D = np.float32([[0,0],[0,550],[550,0],[550,550]])
    # M = cv2.getPerspectiveTransform(Table_2D,affine_table_2D)
    # marked = cv2.warpPerspective(lined,M,(550,550)) # Perspective_Transformation
    cv2.imwrite('JPEGImages/marked' + str(num) + '.jpg', lined)
    return lined, Circle_2D


def circle_trans(Cicle_2D: '上下右左', points_2d: '物体底部四点', lined_img=[]):
    '''
    若最后一个参数非真（调试模式），则显示图片
    '''
    affine_Cicle_2D = np.float32(
        [[300, 0], [300, 600], [600, 300], [0, 300]])  # 圆桌R = 300
    M = cv2.getPerspectiveTransform(Cicle_2D, affine_Cicle_2D)  # 获取透视变换矩阵

    transed_points = np.matmul(points, np.transpose(M))
    for i in range(4):
        transed_points[i][0] = transed_points[i][0] / transed_points[i][2]
        transed_points[i][1] = transed_points[i][1] / transed_points[i][2]

    a = [0 for i in range(4)]
    for i in range(4):
        a[i] = (int(transed_points[i][0]), int(transed_points[i][1]))
    angle = np.degrees(np.arccos(
        (a[0][0] - a[0][1]) / (((a[0][0] - a[0][1])**2 + (a[1][0] - a[1][1])**2) ** 0.5)))

    if len(lined_img):
        # Perspective_Transformation
        transed = cv2.warpPerspective(lined_img, M, (600, 600))
        cv2.line(transed, a[0], a[1], (0, 255, 0), 1)
        cv2.line(transed, a[0], a[2], (0, 255, 0), 1)
        cv2.line(transed, a[1], a[3], (0, 255, 0), 1)
        cv2.line(transed, a[2], a[3], (0, 255, 0), 1)
        cv2.imshow('perspective', transed)

    return transed_points, angle


if __name__ == '__main__':
    ima = cv2.imread("test.jpg", 3)
    out, top, bottle, right, left = my_fun(ima)
    cv2.imshow("output1.jpg",out)
    cv2.waitKey(0)



    