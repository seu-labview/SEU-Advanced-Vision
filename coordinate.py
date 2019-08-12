from cv2 import cv2
import numpy as np
import math
from pyrealsense2 import pyrealsense2 as rs
import time
import png

import corner
from camera import Camera
from yolo6D.utils import get_camera_intrinsic

def getCoordinate(img, depth, M):
    return cv2.reprojectImageTo3D(img, M, depth,)
    

def Depth2World(u, v, depth):
    '''深度图（u,v为像素坐标）转世界坐标'''
    return Pixel2ImagePlane(u, v) * depth


def Pixel2ImagePlane(u, v):
    '''像素坐标转成像平面坐标'''
    fx = 618.33
    fy = 618.33
    ppx = 309.9
    ppy = 237.5
    x =  (u - ppx) / fx
    y =  (v - ppy) / fy
    return (x, y, 1)


def World2Image(wrld):
    '''世界坐标转像素坐标'''
    fx = 618.33
    fy = 618.33
    ppx = 309.9
    ppy = 237.5
    x = wrld[0] / wrld[2]
    y = wrld[1] / wrld[2]
    x = x * fx + ppx
    y = y * fy + ppy
    return (x, y)


def ReadData():
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


if __name__ == '__main__':
    camera = Camera()
    camera.init()
    time.sleep(2)
    depth, img = camera.capture()
    cv2.imwrite('JPEGImages/0.jpg', img)
    cv2.imwrite('depth.jpg', depth)
    x, ca, ho = ReadData()
    lined , Table_2D = corner.square_desk(0, x, ca, ho)
    affine_table_2D = np.float32(
        [[0, 0], [0, 550], [550, 0], [550, 550]])  # 方桌边长550mm
    # M = cv2.getPerspectiveTransform(Table_2D, affine_table_2D)
    # print(getCoordinate(img, depth, M))
    (u, v) = Table_2D[1]
    # print(Depth2World(u, v, depth[int(u), int(v)]))
    rs.rs2_deproject_pixel_to_point(get_camera_intrinsic(), (u, v), depth_of_pixel)
    # From pixel to 3D point
    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)
    print("\n\t 3D depth_point: " + str(depth_point))

    # From 3D depth point to 3D color point
    color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
    print("\n\t 3D color_point: " + str(color_point))

    # From color point to 2D color pixel
    color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
    print("\n\t color_pixel: " + str(color_pixel))