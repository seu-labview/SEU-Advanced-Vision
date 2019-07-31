from pyrealsense2 import pyrealsense2 as rs
import json
import numpy as np

# class Singleton(type):
#     _instances = {}
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]

class Camera():
    # '''单例相机对象'''
    pipeline = rs.pipeline()
    config = rs.config()

    def init(self):
        print(" 6% 开始初始化相机")
        Camera.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        Camera.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        profile = Camera.pipeline.start(Camera.config)
        frames = Camera.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Color Intrinsics 
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                            'ppx': intr.ppx, 'ppy': intr.ppy,
                            'height': intr.height, 'width': intr.width}

        
        with open('intrinsics.json', 'w') as fp:
            json.dump(camera_parameters, fp)

        align_to = rs.stream.color
        Camera.align = rs.align(align_to)
        # T_start = time.time()            
        # FileName = 0

        Camera.frames = Camera.pipeline.wait_for_frames()

    def __del__(self):
        Camera.pipeline.stop()

    def capture(self, num):
        aligned_frames = Camera.align.process(Camera.frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        d = np.asanyarray(aligned_depth_frame.get_data())
        c = np.asanyarray(color_frame.get_data())
        return d, c