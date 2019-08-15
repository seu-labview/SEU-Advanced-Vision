# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
from skimage import morphology

def mydrawpoi(image, point):
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8
    cv2.circle(image, (point[0,0],point[0,1]), point_size, point_color, thickness)


def mythres(img, x):
    '''二值化'''
    red_low = x[0]
    red_high = x[1]
    green_low = x[2]
    green_high = x[3]
    blue_low = x[4]
    blue_high = x[5]
    height, width = img.shape[:2]   #j:高    i:宽
    im_new = np.zeros((height, width), np.uint8)   #粗糙的二值化图像
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

# 深度距离限制 深度图彩色图交 彩色图圆盘阈值 去除小物品 椭圆检测 椭圆最下点 额外三点 空间三点解曲线 相机外参 圆环方程 圆环四点 反投影RGB 圆盘外参 透视变换
if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    device = profile.get_device()
    sensor = device.query_sensors()
    sr = sensor[0]
    sr.set_option(rs.option.motion_range, 220) #深度距离限制 靠这个缩短能看到的最远距离
    sr.set_option(rs.option.accuracy, 3)
    sr.set_option(rs.option.filter_option, 7)
    sr.set_option(rs.option.confidence_threshold, 2)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance_in_meters = 1.4 #深度距离限制 似乎这个距离上限没什么用
    clipping_distance = clipping_distance_in_meters / depth_scale
    align_to = rs.stream.color
    align = rs.align(align_to)
    pc = rs.pointcloud()
    points = rs.points()
    depth_pixel = [320, 240]
    for i in range(30):
        frames = pipeline.wait_for_frames()
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_frame = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())  
        depth_image_meter = depth_image * depth_scale # 想办法将原始深度图像变成以米为单位的深度图
        depth_pixel_value = depth_image_meter[depth_pixel[0],depth_pixel[1]]
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_pixel_value)
        color_image = np.asanyarray(color_frame.get_data())
        ignore_color = 0 #深度距离限制 将看不见的区域变成黑色
        depth_image_3d = np.dstack((depth_image_meter,depth_image_meter,depth_image_meter))
        depth_limited_color_image = np.where((depth_image_3d > clipping_distance_in_meters) | (depth_image_3d <= 0), ignore_color, color_image) #深度距离限制 深度图彩色图交
        thres_value = [150,233,150,233,150,233]
        thres_color_image = mythres(depth_limited_color_image,thres_value)
        kernel = np.ones((3, 3), np.uint8)  
        thres_color_image = cv2.morphologyEx(thres_color_image, cv2.MORPH_CLOSE, kernel)  # 闭运算
        thres_color_image = cv2.morphologyEx(thres_color_image, cv2.MORPH_OPEN, kernel)   # 开运算
        thres_color_image_3d = np.dstack((thres_color_image,thres_color_image,thres_color_image))
        thres_limited_color_image = np.where( (thres_color_image_3d <= 0), ignore_color, color_image)
        contours, hierarchy = cv2.findContours(thres_color_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_id = -1
        cnt__max_area_id = 0
        cnt_max_area = 0
        for cnt in contours:
            cnt_id = cnt_id + 1
            if len(cnt) < 4:
                continue
            S1=cv2.contourArea(cnt)
            if S1 > cnt_max_area:
                cnt_max_area = S1
                cnt__max_area_id = cnt_id
        cnt = contours[cnt__max_area_id]

        for i in cnt:
            mydrawpoi(color_image, i)

        left_point_id = 0
        point_id = -1
        left_point_x = cnt[0,0,0]
        for left_point in cnt:
            point_id = point_id + 1
            if left_point[0,0] < left_point_x:
                left_point_id = point_id
                left_point_x = left_point[0,0]
        left_point_pixel_x = cnt[left_point_id, 0, 0]
        left_point_pixel_y = cnt[left_point_id, 0, 1]
        print('left',left_point_id,left_point_pixel_x,left_point_pixel_y)
        cv2.circle(color_image, (left_point_pixel_x,left_point_pixel_y), 2, (255,0,0), 2)

        right_point_id = 0
        point_id = -1
        right_point_x = cnt[0,0,0]
        for right_point in cnt:
            point_id = point_id + 1
            if right_point[0,0] > right_point_x:
                right_point_id = point_id
                right_point_x = right_point[0,0]
        right_point_pixel_x = cnt[right_point_id, 0, 0]
        right_point_pixel_y = cnt[right_point_id, 0, 1]
        print('right',right_point_id,right_point_pixel_x,right_point_pixel_y)
        cv2.circle(color_image, (right_point_pixel_x,right_point_pixel_y), 2, (255,0,0), 2)


        diff_id = right_point_id - left_point_id
        diff_id = int(diff_id / 3)
        test_1_id = left_point_id + diff_id
        test_1_id_x = cnt[test_1_id, 0, 0]
        test_1_id_y = cnt[test_1_id, 0, 1]
        print('test_1',right_point_id,right_point_pixel_x,right_point_pixel_y)
        cv2.circle(color_image, (test_1_id_x,test_1_id_y), 2, (255,0,0), 2)
        test_2_id = right_point_id - diff_id
        test_2_id_x = cnt[test_2_id, 0, 0]
        test_2_id_y = cnt[test_2_id, 0, 1]
        print('test_2',right_point_id,right_point_pixel_x,right_point_pixel_y)
        cv2.circle(color_image, (test_2_id_x,test_2_id_y), 2, (255,0,0), 2)        

        bottom_point_id = 0
        point_id = -1
        bottom_point_y = cnt[0,0,1]
        for bottom_point in cnt:
            point_id = point_id + 1
            if bottom_point[0,1] > bottom_point_y:
                bottom_point_id = point_id
                bottom_point_y = bottom_point[0,1]
        bottom_point_pixel_x = cnt[bottom_point_id, 0, 0]
        bottom_point_pixel_y = cnt[bottom_point_id, 0, 1]
        print('bottom',bottom_point_id,bottom_point_pixel_x,bottom_point_pixel_y)
        cv2.circle(color_image, (bottom_point_pixel_x,bottom_point_pixel_y), 2, (255,0,0), 2)

        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', color_image)
        cv2.namedWindow('thres_color_image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('thres_color_image', thres_color_image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            cv2.imwrite('depth_limited_color_image.jpg',depth_limited_color_image)
            break
    pipeline.stop()




