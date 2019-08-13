import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import pyrealsense2 as rs
import corner


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

                top = []; bottle = []
                top.append(top_x); top.append(top_y)
                bottle.append(bottle_x); bottle.append(bottle_y)
                
                drawpoi(img, (int(top_x), int(top_y)))
                drawpoi(img, (int(bottle_x), int(bottle_y)))
                drawpoi(img, (int(ell[0][0]), int(ell[0][1])))
                # output, cent_r = imageRotate(output, des_ell[2], (des_ell[1][0], des_ell[1][1])) # 现在桌子转正了
                # drawpoi(output, (int(des_ell[1][0]), int(des_ell[1][1])))

                # drawpoi(output, cent_r)
                # print(cent_r)
                rate = ell[1][1]/ell[1][0]
                # output = imageresize(output, rate) #, cent_s,, cent_r
                # drawpoi(output, cent_s)
                return img, top, bottle
        else:
            top = [0, 0]; bottle = [0, 0]
    #cv2.imshow("output.jpg",output)
    return img, top, bottle
    # canvasPoints = np.float([[97., 35.], [505., 35.] ,[23., 368.], [586., 368.]])
    # srcImg = cv2.
    # perspectiveImg = cv2.warpPerspective(srcImg, perspectiveMatrix, (700, 500))
    # cv2.imshow('PerspectiveImg', perspectiveImg)
    # cv2.waitKey(0)


def Circle_trans(Circle_2D: '圆桌次序：右下左上', corners: '物体底部四点', lined_img=[]):
    '''
    若最后一个参数非真（调试模式），则显示图片
    '''
    affine_table_2D = np.float32(
        [[300, 0], [0, 300], [-300, 0], [0, -300]])  # 圆桌半径300mm
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


if __name__ == '__main__':
    ima = cv2.imread("a.jpg", 3)
    out, top, bottle = my_fun(ima)
    cv2.imshow("output1.jpg",out)
    cv2.waitKey(0)



    