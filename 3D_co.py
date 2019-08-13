import pyrealsense2 as rs
import numpy as np
import cv2
import circle_fit

# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipe_profile = pipeline.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    img_color = np.asanyarray(color_frame.get_data())
    img_depth = np.asanyarray(depth_frame.get_data())

    # Intrinsics & Extrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

    # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Map depth to color
    depth_pixel = [240, 320]   # Random pixel
    # depth_pixel = [0,100]
    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)

    color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
    color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
    # print ('depth: ',color_point)
    # print ('depth: ',color_pixel)

    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    tex = np.asanyarray(points.get_texture_coordinates())
    # i = 640*200+200
    img_color, top, bottle = circle_fit.my_fun(img_color)
    # i = 640 * int(top[1]) + int(top[0])
    # print("i = ", i)

    i = 640 * int(top[1]) + int(top[0])
    print("i = ", i)
    if top[1] > 0 and top[1] < 640 and top[0] > 0 and top[0] < 480 and i < 300000:
        print ('top: ',[np.float(vtx[i][0]),np.float(vtx[i][1]),np.float(vtx[i][2])])
        cv2.circle(img_color, (int(top[0]),int(top[1])), 8, [255,0,255], thickness=-1)
        cv2.putText(img_color,"Dis:"+str(img_depth[int(top[0]),int(top[1])]), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])
        cv2.putText(img_color,"X:"+str(np.float(vtx[i][0])), (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])
        cv2.putText(img_color,"Y:"+str(np.float(vtx[i][1])), (80,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])
        cv2.putText(img_color,"Z:"+str(np.float(vtx[i][2])), (80,160), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])


    j = 640 * int(bottle[1]) + int(bottle[0])
    print("j = ", j)
    if bottle[1] > 0 and bottle[1] < 640 and bottle[0] > 0 and bottle[0] < 480 and j < 300000:
        print ('bottle: ',[np.float(vtx[j][0]),np.float(vtx[j][1]),np.float(vtx[j][2])])
        cv2.circle(img_color, (int(bottle[0]),int(bottle[1])), 8, [255,0,255], thickness=-1)
        cv2.putText(img_color,"Dis:"+str(img_depth[int(bottle[0]),int(bottle[1])]), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])
        cv2.putText(img_color,"X:"+str(np.float(vtx[j][0])), (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])
        cv2.putText(img_color,"Y:"+str(np.float(vtx[j][1])), (80,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])
        cv2.putText(img_color,"Z:"+str(np.float(vtx[j][2])), (80,160), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])




    cv2.imshow('depth_frame',img_color)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    '''
    npy_vtx = np.zeros((len(vtx), 3), float)
    print('len: ',len(vtx))
    for i in range(len(vtx)):
        npy_vtx[i][0] = np.float(vtx[i][0])
        npy_vtx[i][1] = np.float(vtx[i][1])
        npy_vtx[i][2] = np.float(vtx[i][2])

    npy_tex = np.zeros((len(tex), 3), float)
    for i in range(len(tex)):
        npy_tex[i][0] = np.float(tex[i][0])
        npy_tex[i][1] = np.float(tex[i][1])
    '''

pipeline.stop()

'''
pc = rs.pointcloud()
frames = pipeline.wait_for_frames()
depth = frames.get_depth_frame()
color = frames.get_color_frame()
img_color = np.asanyarray(color_frame.get_data())
img_depth = np.asanyarray(depth_frame.get_data())
pc.map_to(color)
points = pc.calculate(depth)
vtx = np.asanyarray(points.get_vertices())
tex = np.asanyarray(points.get_texture_coordinates())

npy_vtx = np.zeros((len(vtx), 3), float)
for i in range(len(vtx)):
    npy_vtx[i][0] = np.float(vtx[i][0])
    npy_vtx[i][1] = np.float(vtx[i][1])
    npy_vtx[i][2] = np.float(vtx[i][2])

npy_tex = np.zeros((len(tex), 3), float)
for i in range(len(tex)):
    npy_tex[i][0] = np.float(tex[i][0])
    npy_tex[i][1] = np.float(tex[i][1])
'''
