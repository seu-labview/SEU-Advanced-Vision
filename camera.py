from pyrealsense2 import pyrealsense2 as rs
import json
import numpy as np

class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance

class Camera(Singleton):
    '''单例相机对象'''
    pipeline = rs.pipeline()
    config = rs.config()
    align = None

    def __new__(self):
        # global pipeline
        # global config
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
        # global align
        Camera.align = rs.align(align_to)
        # T_start = time.time()            
        # FileName = 0

    def __del__(self):
        Camera.pipeline.stop()

    def capture(self, num):
        frames = Camera.pipeline.wait_for_frames()
        aligned_frames = Camera.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        d = np.asanyarray(aligned_depth_frame.get_data())
        c = np.asanyarray(color_frame.get_data())
        return d, c