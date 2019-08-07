import numpy as np
from cv2 import cv2
import os
import math
import sys
from yolo6D.utils import get_camera_intrinsic
from yolo6D.Predict import draw

# def draw(img, corner, imgpts):
#     img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
#     cv2.putText(img, "X",tuple(imgpts[0].ravel()),cv2.FONT_HERSHEY_COMPLEX,0.5,(100,149,237),2)
#     img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
#     cv2.putText(img, "Y",tuple(imgpts[1].ravel()),cv2.FONT_HERSHEY_COMPLEX,0.5,(200,20,127),2)
#     img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
#     cv2.putText(img, "-Z",tuple(imgpts[2].ravel()),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,140,0),2)
#     return img

#open file and read picture
def read_data_cfg(datacfg):
    options = dict()
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options

def getcrosspoint(rho1,theta1,rho2,theta2):
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
    print(corss_x,cross_y)
    return (corss_x,cross_y)

def square_desk(num):
    '''
    输入：图片编号
    输出：坐标系
    导出：带坐标系图片
    '''
    img_path = 'JPEGImages/predict' + str(num) + '.jpg'
    img = cv2.imread(img_path,1)
    img = cv2.resize(img,(640,480), interpolation = cv2.INTER_CUBIC)

    internal_calibration = get_camera_intrinsic()
    internal_calibration = np.array(internal_calibration, dtype='float32')
    distCoeffs = np.zeros((8, 1), dtype='float32')

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # threshold
    lower_gray = np.array([140])
    upper_gray = np.array([200])
    test_gray = cv2.inRange(gray,lower_gray,upper_gray)
    kernel = np.ones((3,3),np.uint8)
    test_gray = cv2.morphologyEx(test_gray, cv2.MORPH_CLOSE, kernel)
    test_gray = cv2.morphologyEx(test_gray, cv2.MORPH_OPEN, kernel)
    kernel2 = np.ones((5,5),np.uint8)
    test_gray = cv2.morphologyEx(test_gray, cv2.MORPH_CLOSE, kernel2)
    edges = cv2.Canny(test_gray,100,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,80)
    top_line_theta = []
    top_line_rho = []
    left_line_theta = []
    left_line_rho = []
    right_line_theta = []
    right_line_rho = []
    bottom_line_theta = []
    bottom_line_rho = []
    summ = 0
    final_lines = np.zeros((4,2))
    if lines is None:
        print("    \033[0;31m未检测到方桌！\033[0m")
    else:
        for line in lines:
            for rho,theta in line:

                if theta > (math.pi*(2/3)):
                    if abs(rho) <320:
                        right_line_theta.append(theta)
                        right_line_rho.append(rho)
                elif theta > (math.pi/4):
                    if abs(rho) <320:
                        top_line_theta.append(theta)
                        top_line_rho.append(rho)
                    else:
                        bottom_line_theta.append(theta)
                        bottom_line_rho.append(rho)
                else:
                    left_line_theta.append(theta)
                    left_line_rho.append(rho)
            

        for i in right_line_theta:
            summ +=i
        right_line_theta_average = summ / len(right_line_theta)
        final_lines[0,1] = right_line_theta_average
        summ = 0
        for i in right_line_rho:
            summ +=i
        right_line_rho_average = summ / len(right_line_rho)
        final_lines[0,0] = right_line_rho_average
        summ = 0 

        for i in left_line_theta:
            summ +=i
        left_line_theta_average = summ / len(left_line_theta)
        final_lines[1,1] = left_line_theta_average
        summ = 0
        for i in left_line_rho:
            summ +=i
        left_line_rho_average = summ / len(left_line_rho)
        final_lines[1,0] = left_line_rho_average
        summ = 0

        for i in top_line_theta:
            summ +=i
        top_line_theta_average = summ / len(top_line_theta)
        final_lines[2,1] = top_line_theta_average
        summ = 0
        for i in top_line_rho:
            summ +=i
        top_line_rho_average = summ / len(top_line_rho)
        final_lines[2,0] = top_line_rho_average
        summ = 0

        for i in bottom_line_theta:
            summ +=i
        bottom_line_theta_average = summ / len(bottom_line_theta)
        final_lines[3,1] = bottom_line_theta_average
        summ = 0
        for i in bottom_line_rho:
            summ +=i
        bottom_line_rho_average = summ / len(bottom_line_rho)
        final_lines[3,0] = bottom_line_rho_average
        summ = 0
        print(final_lines)
        final_lines = np.array(final_lines)
        for i in range(4):
            theta = final_lines[i,1]
            rho = final_lines[i,0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),(200,135,100),2)
            string = str(i)
            cv2.putText(img, string,(int(x0),int(y0)),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1)
        
        left_top_point_x, left_top_point_y = getcrosspoint(left_line_rho_average,left_line_theta_average,top_line_rho_average,top_line_theta_average)
        left_bottom_point_x, left_bottom_point_y = getcrosspoint(left_line_rho_average,left_line_theta_average,bottom_line_rho_average,bottom_line_theta_average) 
        right_top_point_x, right_top_point_y = getcrosspoint(right_line_rho_average,right_line_theta_average,top_line_rho_average,top_line_theta_average)
        right_bottom_point_x, right_bottom_point_y = getcrosspoint(right_line_rho_average,right_line_theta_average,bottom_line_rho_average,bottom_line_theta_average)
        
        Table_2D = []
        Table_2D.append([left_top_point_x, left_top_point_y])
        Table_2D.append([left_bottom_point_x, left_bottom_point_y])
        Table_2D.append([right_top_point_x, right_top_point_y])
        Table_2D.append([right_bottom_point_x, right_bottom_point_y])
        cv2.circle(img, (left_top_point_x, left_top_point_y), 3, (255,0,0),-1)
        cv2.circle(img, (left_bottom_point_x, left_bottom_point_y), 3, (255,0,0),-1)
        cv2.circle(img, (right_top_point_x, right_top_point_y), 3, (255,0,0),-1)
        cv2.circle(img, (right_bottom_point_x, right_bottom_point_y), 3, (255,0,0),-1)
        Table_3D = []
        Table_3D.append([0,0,0])
        Table_3D.append([0,55,0])
        Table_3D.append([55,0,0])
        Table_3D.append([55,55,0])
        Table_3D = np.array(Table_3D,dtype='float32')
        Table_2D = np.array(Table_2D,dtype='float32')
        _,rvector,tvector=cv2.solvePnP(Table_3D,Table_2D,internal_calibration,distCoeffs)
        axis = np.float32([[55,0,0], [0,55,0], [0,0,-20]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis, rvector, tvector,internal_calibration,distCoeffs,)
        img = draw(img,(left_top_point_x, left_top_point_y),imgpts)

    cv2.imwrite('corner.jpg',img)