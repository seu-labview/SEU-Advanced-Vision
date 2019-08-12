import numpy as np
from cv2 import cv2
import os
import math
import sys
from yolo6D.utils import get_camera_intrinsic
from yolo6D.Predict import draw
from Kmeans import Cluster
import copy
from skimage import morphology

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

def square_canny(img, canny):
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # threshold
    # lower_gray = np.array([140])
    # upper_gray = np.array([200])
    # test_gray = cv2.inRange(gray,lower_gray,upper_gray)
    kernel = np.ones((9, 9), np.uint8)
    test_gray = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    test_gray = cv2.morphologyEx(test_gray, cv2.MORPH_OPEN, kernel)
    kernel2 = np.ones((9, 9), np.uint8)
    test_gray = cv2.morphologyEx(test_gray, cv2.MORPH_CLOSE, kernel2)
    edges = cv2.Canny(test_gray, canny[0], canny[1], canny[2])
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
        print("    \033[0;31m未检测到方桌！\033[0m")
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

def square_desk(num, x, canny, hough):
    '''
    输入：图片编号
    输出：坐标系
    导出：带坐标系图片
    '''
    img_path = 'JPEGImages/' + str(num) + '.jpg'
    img = cv2.imread(img_path, 1)

    thresed = thres(img, x)
    thresed_new = remove_small_objects(thresed)
    edges = square_canny(thresed_new, canny)
    lined, Table_2D = square_line(img, edges, hough)
    # affine_table_2D = np.float32([[0,0],[0,550],[550,0],[550,550]])
    # M = cv2.getPerspectiveTransform(Table_2D,affine_table_2D)
    # marked = cv2.warpPerspective(lined,M,(550,550)) # Perspective_Transformation
    cv2.imwrite('JPEGImages/marked' + str(num) + '.jpg', lined)
    return lined, Table_2D

def square_trans(Table_2D: '桌子四角', corners: '物体底部四点', lined_img=[]):
    '''
    若最后一个参数非真（调试模式），则显示图片
    '''
    affine_table_2D = np.float32(
        [[0, 0], [0, 550], [550, 0], [550, 550]])  # 方桌边长550mm
    M = cv2.getPerspectiveTransform(Table_2D, affine_table_2D)
    # a3x3 = np.resize(np.append(affine_table_2D, [[1], [1], [1], [1]], axis=1), (3,3))
    # t3x3 = np.resize(np.append(Table_2D, [[1],[1],[1],[1]], axis = 1), (3,3))

    transed_corners = np.matmul(corners, np.transpose(M))
    for i in range(4):
        transed_corners[i][0] = transed_corners[i][0] / transed_corners[i][2]
        transed_corners[i][1] = transed_corners[i][1] / transed_corners[i][2]

    a = [0 for i in range(4)]
    for i in range(4):
        a[i] = (int(transed_corners[i][0]), int(transed_corners[i][1]))
    angle = np.degrees(np.arccos(
        (a[0][0] - a[0][1]) / (((a[0][0] - a[0][1])**2 + (a[1][0] - a[1][1])**2) ** 0.5)))

    if len(lined_img):
        # Perspective_Transformation
        transed = cv2.warpPerspective(lined_img, M, (550, 550))
        cv2.line(transed, a[0], a[1], (0, 255, 0), 1)
        cv2.line(transed, a[0], a[2], (0, 255, 0), 1)
        cv2.line(transed, a[1], a[3], (0, 255, 0), 1)
        cv2.line(transed, a[2], a[3], (0, 255, 0), 1)
        cv2.imshow('perspective', transed)

    return transed_corners, angle
