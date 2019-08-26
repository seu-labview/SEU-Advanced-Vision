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
    for i in range(30):
        frames = pipeline.wait_for_frames()
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        depth_intrinsic = [[618.329,0,309.857],[0,618.329,237.485],[0,0,1]]
        color_frame = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())  
        depth_image_meter = depth_image * depth_scale # 想办法将原始深度图像变成以米为单位的深度图
        color_image = np.asanyarray(color_frame.get_data())
        
        ignore_color = 0 #深度距离限制 将看不见的区域变成黑色
        depth_image_3d = np.dstack((depth_image_meter,depth_image_meter,depth_image_meter))
        depth_limited_color_image = np.where((depth_image_3d > clipping_distance_in_meters) | (depth_image_3d <= 0), ignore_color, color_image) #深度距离限制 深度图彩色图交
        thres_value = [150,255,155,255,150,255]
        thres_color_image = mythres(depth_limited_color_image,thres_value)
        kernel = np.ones((3, 3), np.uint8)  
        thres_color_image = cv2.morphologyEx(thres_color_image, cv2.MORPH_CLOSE, kernel)  # 闭运算
        thres_color_image = cv2.morphologyEx(thres_color_image, cv2.MORPH_OPEN, kernel)   # 开运算
        thres_color_image = cv2.erode(thres_color_image, kernel, iterations = 1)
        thres_color_image = cv2.erode(thres_color_image, kernel, iterations = 1)
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

        depth_pixel = []
        depth_pixel_value = []

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
        depth_pixel.append([left_point_pixel_x,left_point_pixel_y])
        depth_pixel_value.append(depth_image_meter[left_point_pixel_y,left_point_pixel_x])
        # print('left',left_point_id,left_point_pixel_x,left_point_pixel_y)
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
        depth_pixel.append([right_point_pixel_x,right_point_pixel_y])
        depth_pixel_value.append(depth_image_meter[right_point_pixel_y,right_point_pixel_x])
        # print('right',right_point_id,right_point_pixel_x,right_point_pixel_y)
        cv2.circle(color_image, (right_point_pixel_x,right_point_pixel_y), 2, (255,0,0), 2)


        diff_id = right_point_id - left_point_id
        diff_id = int(diff_id / 3)
        test_1_id = left_point_id + diff_id
        test_1_id_x = cnt[test_1_id, 0, 0]
        test_1_id_y = cnt[test_1_id, 0, 1]
        depth_pixel.append([test_1_id_x,test_1_id_y])
        depth_pixel_value.append(depth_image_meter[test_1_id_y,test_1_id_x])
        # print('test_1',right_point_id,right_point_pixel_x,right_point_pixel_y)
        cv2.circle(color_image, (test_1_id_x,test_1_id_y), 2, (255,0,0), 2)
        test_2_id = right_point_id - diff_id
        test_2_id_x = cnt[test_2_id, 0, 0]
        test_2_id_y = cnt[test_2_id, 0, 1]
        depth_pixel.append([test_2_id_x,test_2_id_y])
        depth_pixel_value.append(depth_image_meter[test_2_id_y,test_2_id_x])
        # print('test_2',right_point_id,right_point_pixel_x,right_point_pixel_y)
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
        depth_pixel.append([bottom_point_pixel_x,bottom_point_pixel_y])
        depth_pixel_value.append(depth_image_meter[bottom_point_pixel_y,bottom_point_pixel_x])
        # print('bottom',bottom_point_id,bottom_point_pixel_x,bottom_point_pixel_y)
        cv2.circle(color_image, (bottom_point_pixel_x,bottom_point_pixel_y), 2, (255,0,0), 2)
        # print(depth_pixel)
        # print(depth_pixel_value)
        depth_camera_3D_point = []
        for i in range(5):
            depth_camera_3D_point.append(rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel[i],depth_pixel_value[i] )) 
        # print(depth_camera_3D_point)
        depth_camera_3D_point = np.array(depth_camera_3D_point, dtype='float32')
        depth_pixel = np.array(depth_pixel, dtype='float32')
        depth_intrinsic = np.array(depth_intrinsic, dtype='float32')
        distCoeffs = np.zeros((8, 1), dtype='float32')
        _, rvector, tvector = cv2.solvePnP(
            depth_camera_3D_point, depth_pixel, depth_intrinsic, distCoeffs)
        left_3D_point = depth_camera_3D_point[0]
        right_3D_point = depth_camera_3D_point[1]
        bottom_3D_point = depth_camera_3D_point[4]
        vector1 = left_3D_point - bottom_3D_point
        vector2 = right_3D_point - bottom_3D_point
        vertical_vector = np.cross(vector1,vector2)
        vertical_vector_norm = np.linalg.norm(vertical_vector)
        unit_vector = vertical_vector / vertical_vector_norm
        vertical_point = bottom_3D_point + vertical_vector
        D1 = np.dot(bottom_3D_point,vertical_vector)
        middle_point1 = (left_3D_point + bottom_3D_point) / 2
        middle_point2 = (right_3D_point + bottom_3D_point) / 2
        D2 = np.dot(middle_point1,vector1)
        D3 = np.dot(middle_point2,vector2)
        D_equation_of_center_point = np.array([D1,D2,D3],dtype='float32')
        A_equation_of_center_point = [vertical_vector,vector1,vector2]
        A_equation_of_center_point = np.array(A_equation_of_center_point, dtype='float32')
        center_point = np.linalg.solve(A_equation_of_center_point, D_equation_of_center_point)
        vecter_x1 = center_point - bottom_3D_point
        top_x_point = center_point + vecter_x1
        vecter_y1 = np.cross(vecter_x1,unit_vector)
        side_y1_point = center_point + vecter_y1
        side_y2_point = center_point - vecter_y1
        # print(center_point)
        propoint = []
        propoint.append(vertical_point)
        propoint.append(center_point)
        propoint.append(top_x_point)
        propoint.append(side_y1_point)
        propoint.append(side_y2_point)
        propoint = np.array(propoint, dtype='float32')
        imgpts, _ = cv2.projectPoints(propoint, rvector, tvector, depth_intrinsic, distCoeffs)
        cv2.line(color_image, (bottom_point_pixel_x,bottom_point_pixel_y), tuple(imgpts[0].ravel()), (200, 55, 100), 3)
        cv2.line(color_image, (bottom_point_pixel_x,bottom_point_pixel_y), (right_point_pixel_x,right_point_pixel_y), (200, 55, 100), 3)
        cv2.line(color_image, (bottom_point_pixel_x,bottom_point_pixel_y), (left_point_pixel_x,left_point_pixel_y), (200, 55, 100), 3)
        cv2.circle(color_image, tuple(imgpts[1].ravel()), 2, (100,200,50), 2)
        cv2.circle(color_image, tuple(imgpts[2].ravel()), 2, (100,200,50), 2)
        cv2.circle(color_image, tuple(imgpts[3].ravel()), 2, (100,200,50), 2)
        cv2.circle(color_image, tuple(imgpts[4].ravel()), 2, (100,200,50), 2)
        circle_pixel = [[bottom_point_pixel_x,bottom_point_pixel_y],imgpts[2].ravel(),imgpts[3].ravel(),imgpts[4].ravel()]
        circle_pixel = np.array(circle_pixel,dtype = 'float32')
        print(circle_pixel)
        affine_circle_pixel = np.float32([[600, 300], [0, 300], [300, 0], [300, 600]])
        M = cv2.getPerspectiveTransform(circle_pixel, affine_circle_pixel)
        Perspective_image = cv2.warpPerspective(color_image,M,(600,600))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', color_image)
        cv2.namedWindow('thres_color_image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('thres_color_image', thres_color_image)
        cv2.namedWindow('Perspective', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Perspective', Perspective_image)        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            cv2.imwrite('depth_limited_color_image.jpg',depth_limited_color_image)
            break
    pipeline.stop()




