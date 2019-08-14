from pyrealsense2 import pyrealsense2 as rs
import json
import numpy as np
from retrying import retry
import os

# class Singleton(type):
#     _instances = {}
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]


class Camera():
    '''相机函数封装'''
    pipeline = rs.pipeline()
    config = rs.config()

    def init(self):
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start pipeline
        attempt = 1  # 尝试次数，成功则为-1
        while(attempt > 0):
            try:
                profile = self.pipeline.start(self.config)
            except(RuntimeError):
                print("\033[0;31m未检测到相机！重试%s/5\033[0m" % (attempt))
                attempt += 1
                if attempt > 5:
                    os._exit(0)
            else:
                attempt = 0
        while(attempt >= 0):
            try:
                self.frames = self.pipeline.wait_for_frames()
            except(RuntimeError):
                print("\033[0;31m相机卡死！重试%s/5\033[0m" % attempt)
                attempt += 1
                if attempt > 5:
                    os._exit(0)
            else:
                attempt = -1
        color_frame = self.frames.get_color_frame()
        depth_frame = self.frames.get_depth_frame()
        device = profile.get_device()
        sensor = device.query_sensors()
        sr = sensor[0]
        sr.set_option(rs.option.motion_range, 140)
        sr.set_option(rs.option.accuracy, 3)
        sr.set_option(rs.option.filter_option, 5)
        sr.set_option(rs.option.confidence_threshold, 15)

        # Color Intrinsics
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                             'ppx': intr.ppx, 'ppy': intr.ppy,
                             'height': intr.height, 'width': intr.width}
        with open('intrinsics.json', 'w') as fp:
            json.dump(camera_parameters, fp)

        depth_intr = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_parameters = {'fx': depth_intr.fx, 'fy': depth_intr.fy,
                            'ppx': depth_intr.ppx, 'ppy': depth_intr.ppy,
                            'height': depth_intr.height, 'width': depth_intr.width}
        with open('depth_intrinsics.json', 'w') as fp:
            json.dump(depth_parameters, fp)

        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
            color_frame.profile)
        extrin_parameters = {'rot': depth_to_color_extrin.rotation,
                             'tran': depth_to_color_extrin.translation}
        with open('extrinsics.json', 'w') as fp:
            json.dump(extrin_parameters, fp)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        for i in range(30):
            self.frames = self.pipeline.wait_for_frames()

    def __del__(self):
        self.pipeline.stop()

    def capture(self):
        '''返回：深度图，彩色图'''
        attempt = 1
        while(attempt >= 0):
            try:
                self.frames = self.pipeline.wait_for_frames()
            except(RuntimeError):
                print("\033[0;31m相机卡死！重试%s/5\033[0m" % attempt)
                attempt += 1
                if attempt > 5:
                    os._exit(0)
            else:
                attempt = -1
        
        aligned_frames = self.align.process(self.frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        d = np.asanyarray(aligned_depth_frame.get_data())
        c = np.asanyarray(color_frame.get_data())
        return d, c
