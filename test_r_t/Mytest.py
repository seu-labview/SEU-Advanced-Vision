import numpy as np
import cv2
import os
from MeshPly import *
import math
def get_camera_intrinsic():
    K = np.zeros((3, 3), dtype='float64')
    K[0, 0], K[0, 2] = 618.3287, 309.8568
    K[1, 1], K[1, 2] = 618.3289, 237.4846
    K[2, 2] = 1.
    return K

def get_3D_corners(vertices):
    
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
    return corners

def draw(img, corner, imgpts):
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
    cv2.putText(img, "X",tuple(imgpts[0].ravel()),cv2.FONT_HERSHEY_COMPLEX,0.5,(100,149,237),2)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    cv2.putText(img, "Y",tuple(imgpts[1].ravel()),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,127),2)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
    cv2.putText(img, "Z",tuple(imgpts[2].ravel()),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,140,0),2)
    return img

internal_calibration = get_camera_intrinsic()
meshname = 'phone.ply'
name_com = '000000'
img_name = name_com + '.jpg'
path_name = name_com + '.txt'

img = cv2.imread(img_name,1)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
k = cv2.waitKey(0)
labpath = path_name
width = 640
height = 480
x_pose = []
y_pose = []
a = np.arange(18)
corners2D_gt = a.reshape(9, 2)
if os.path.getsize(labpath):
    bs = np.loadtxt(labpath)
    x0 = bs[1]  
    y0 = bs[2]
    x1 = bs[3]
    y1 = bs[4]
    x2 = bs[5]
    y2 = bs[6]
    x3 = bs[7]
    y3 = bs[8]
    x4 = bs[9]
    y4 = bs[10]
    x5 = bs[11]
    y5 = bs[12]
    x6 = bs[13]
    y6 = bs[14]
    x7 = bs[15]
    y7 = bs[16]
    x8 = bs[17]
    y8 = bs[18]
x_pose.append(int(x0 * width))
x_pose.append(int(x1 * width))
x_pose.append(int(x2 * width))
x_pose.append(int(x3 * width))
x_pose.append(int(x4 * width))
x_pose.append(int(x5 * width))
x_pose.append(int(x6 * width))
x_pose.append(int(x7 * width))
x_pose.append(int(x8 * width))

y_pose.append(int(y0 * height))  
y_pose.append(int(y1 * height))  
y_pose.append(int(y2 * height))  
y_pose.append(int(y3 * height))  
y_pose.append(int(y4 * height))  
y_pose.append(int(y5 * height))  
y_pose.append(int(y6 * height))
y_pose.append(int(y7 * height))  
y_pose.append(int(y8 * height))


for i in range(9):
    cv2.circle(img, (x_pose[i],y_pose[i]), 1, (255,0,255),-1)
    string = str(i)
    cv2.putText(img, string,(x_pose[i],y_pose[i]),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1)
    corners2D_gt[i,0] = x_pose[i]
    corners2D_gt[i,1] = y_pose[i]


cv2.line(img,(x_pose[1],y_pose[1]),(x_pose[2],y_pose[2]),(255,255,0),2)
cv2.line(img,(x_pose[2],y_pose[2]),(x_pose[4],y_pose[4]),(255,255,0),2)
cv2.line(img,(x_pose[3],y_pose[3]),(x_pose[4],y_pose[4]),(255,255,0),2)
cv2.line(img,(x_pose[1],y_pose[1]),(x_pose[3],y_pose[3]),(255,255,0),2)

cv2.line(img,(x_pose[1],y_pose[1]),(x_pose[5],y_pose[5]),(255,255,0),2)
cv2.line(img,(x_pose[2],y_pose[2]),(x_pose[6],y_pose[6]),(255,255,0),2)
cv2.line(img,(x_pose[3],y_pose[3]),(x_pose[7],y_pose[7]),(255,255,0),2)
cv2.line(img,(x_pose[4],y_pose[4]),(x_pose[8],y_pose[8]),(255,255,0),2)

cv2.line(img,(x_pose[5],y_pose[5]),(x_pose[6],y_pose[6]),(255,255,0),2)
cv2.line(img,(x_pose[5],y_pose[5]),(x_pose[7],y_pose[7]),(255,255,0),2)
cv2.line(img,(x_pose[6],y_pose[6]),(x_pose[8],y_pose[8]),(255,255,0),2)
cv2.line(img,(x_pose[7],y_pose[7]),(x_pose[8],y_pose[8]),(255,255,0),2)

distCoeffs = np.zeros((8, 1), dtype='float32')
mesh          = MeshPly(meshname)
vertices      = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
corners3D     = get_3D_corners(vertices)

#rvector = np.array((1,3),dtype='float32')
#tvector = np.array((1,3),dtype='float32')
objpoints3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32')
KK = np.array(internal_calibration, dtype='float32')
corners2D_gt = np.array(corners2D_gt, dtype='float32')


retval,rvector,tvector,inlier=cv2.solvePnPRansac(objpoints3D,corners2D_gt,KK,distCoeffs)
print(rvector)
print(tvector)
print(inlier)
rmatrix,_ =cv2.Rodrigues(rvector)
print(rmatrix)
theta_x = math.atan2(rmatrix[2,1], rmatrix[2,2])
theta_y = math.atan2(-rmatrix[2,0], math.sqrt(rmatrix[2,1] ** 2 + rmatrix[2,2] ** 2))
theta_z = math.atan2(rmatrix[1,0], rmatrix[0,0])
du_x = (theta_x / math.pi) * 180
du_y = (theta_y / math.pi) * 180
du_z = (theta_z / math.pi) * 180
print(du_x,du_y,du_z)
axis = np.float32([[0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)


imgpts, _ = cv2.projectPoints(axis, rvector, tvector,KK,distCoeffs,)
print(imgpts)
img = draw(img,(x_pose[0],y_pose[0]),imgpts)
if k == 27:         # wait for ESC key to exit5
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('labelmask000000.jpg',img)
    cv2.destroyAllWindows()   
