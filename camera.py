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
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        attempt = 1 # 尝试次数，成功则为-1
        while(attempt > 0):
            try:
                profile = self.pipeline.start(self.config)
            except(RuntimeError):
                print("    \033[0;31m未检测到相机！重试%s/5\033[0m" % (attempt))
                attempt += 1
                if attempt > 5:
                    os._exit(0)
            else:
                attempt = 0
        while(attempt >= 0):
            try:
                frames = self.pipeline.wait_for_frames()
            except(RuntimeError):
                print("    \033[0;31m相机卡死！重试%s/5\033[0m" % attempt)
                attempt += 1
                if attempt > 5:
                    os._exit(0)
            else:
                attempt = -1
        color_frame = frames.get_color_frame()

        # Color Intrinsics 
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                            'ppx': intr.ppx, 'ppy': intr.ppy,
                            'height': intr.height, 'width': intr.width}

        
        with open('intrinsics.json', 'w') as fp:
            json.dump(camera_parameters, fp)

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        # T_start = time.time()            
        # FileName = 0

        self.frames = self.pipeline.wait_for_frames()

    def __del__(self):
        self.pipeline.stop()

    def capture(self, num):
        self.frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(self.frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        d = np.asanyarray(aligned_depth_frame.get_data())
        c = np.asanyarray(color_frame.get_data())
        return d, c