import numpy as np
import cv2
from cv2 import cv2 as cv
import os
import math
import os
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io

import darknet
from darknet import Darknet as dn
import dataset
from utils import get_camera_intrinsic, do_detect, get_3D_corners#, corner_confidence9
from MeshPly import MeshPly
'''
def draw(img, corner, imgpts):
'''#绘制坐标系
'''
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
    cv.putText(img, "X",tuple(imgpts[0].ravel()),cv.FONT_HERSHEY_COMPLEX,0.5,(100,149,237),2)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    cv.putText(img, "Y",tuple(imgpts[1].ravel()),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,127),2)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
    cv.putText(img, "Z",tuple(imgpts[2].ravel()),cv.FONT_HERSHEY_COMPLEX,0.5,(255,140,0),2)
    return img
'''

'''
def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    boxes = dn.make_boxes(net)
    probs = dn.make_probs(net)
    num =   dn.num_boxes(net)
    dn.network_detect(net, image, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    dn.free_ptrs(dn.cast(probs, dn.POINTER(dn.c_void_p)), num)
    return res
'''

def makedirs(path):
    '''Create new directory'''
    if not os.path.exists( path ):
        os.makedirs( path )

# def valid(datacfg, cfgfile, weightfile, outfile):
def detect(name, cfgfile, weightfile, image_path):
    '''
    调用神经网络检测
    输入：图片位置
    返回：10个二维点坐标组成的2*10数组
    '''

    # Parameters
    seed         = int(time.time())
    gpus         = '0'     # Specify which gpus to use
    torch.manual_seed(seed)
    use_cuda = False
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    model = dn(cfgfile)
    model.print_network()
    model.load_weights(weightfile)
    img = cv.imread(image_path,1)
    return do_detect(model, img, 0.1, 0.4, 0)
    

def predict(name, num):
    '''
    ！注意：需将权重文件放在 yolo6D 下
    单物体预测并框出
    输入: 物体名称，图片编号
    导出：带框图片
    '''
    # folder = 'LINEMOD/' + name + '/'
    # internal_calibration = get_camera_intrinsic()
    # meshname = folder + 'registeredScene.ply'
    img_name = num + '.jpg'
    # path_name = num + '.txt'

    img = cv.imread(img_name,1)

    boxes = detect(name, 'yolo-pose.cfg', '/home/scirocco/' + name + '.weights', num + '.jpg')
    best_conf_est = -1
    for j in range(len(boxes)):
        if (boxes[j][18] > best_conf_est):
            box_pr        = boxes[j]
            best_conf_est = boxes[j][18]

    bs = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
    # print("\nbs:\n")
    # print(bs)
    # print("\n\n")
    # labpath = path_name
    width = 640
    height = 480
    x_pose = []
    y_pose = []
    a = np.arange(18)
    corners2D_gt = a.reshape(9, 2)

    x0 = bs[0][0]  
    y0 = bs[0][1]
    x1 = bs[1][0]
    y1 = bs[1][1]
    x2 = bs[2][0]
    y2 = bs[2][1]
    x3 = bs[3][0]
    y3 = bs[3][1]
    x4 = bs[4][0]
    y4 = bs[4][1]
    x5 = bs[5][0]
    y5 = bs[5][1]
    x6 = bs[6][0]
    y6 = bs[6][1]
    x7 = bs[7][0]
    y7 = bs[7][1]
    x8 = bs[8][0]
    y8 = bs[8][1]
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
        cv.circle(img, (x_pose[i],y_pose[i]), 1, (255,0,255),-1)
        string = str(i)
        cv.putText(img, string,(x_pose[i],y_pose[i]),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1)
        corners2D_gt[i,0] = x_pose[i]
        corners2D_gt[i,1] = y_pose[i]


    cv.line(img,(x_pose[1],y_pose[1]),(x_pose[2],y_pose[2]),(255,255,0),2)
    cv.line(img,(x_pose[2],y_pose[2]),(x_pose[4],y_pose[4]),(255,255,0),2)
    cv.line(img,(x_pose[3],y_pose[3]),(x_pose[4],y_pose[4]),(255,255,0),2)
    cv.line(img,(x_pose[1],y_pose[1]),(x_pose[3],y_pose[3]),(255,255,0),2)

    cv.line(img,(x_pose[1],y_pose[1]),(x_pose[5],y_pose[5]),(255,255,0),2)
    cv.line(img,(x_pose[2],y_pose[2]),(x_pose[6],y_pose[6]),(255,255,0),2)
    cv.line(img,(x_pose[3],y_pose[3]),(x_pose[7],y_pose[7]),(255,255,0),2)
    cv.line(img,(x_pose[4],y_pose[4]),(x_pose[8],y_pose[8]),(255,255,0),2)

    cv.line(img,(x_pose[5],y_pose[5]),(x_pose[6],y_pose[6]),(255,255,0),2)
    cv.line(img,(x_pose[5],y_pose[5]),(x_pose[7],y_pose[7]),(255,255,0),2)
    cv.line(img,(x_pose[6],y_pose[6]),(x_pose[8],y_pose[8]),(255,255,0),2)
    cv.line(img,(x_pose[7],y_pose[7]),(x_pose[8],y_pose[8]),(255,255,0),2)

    # distCoeffs = np.zeros((8, 1), dtype='float32')
    # mesh          = MeshPly(meshname)
    # vertices      = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    # corners3D     = get_3D_corners(vertices)

    #rvector = np.array((1,3),dtype='float32')
    #tvector = np.array((1,3),dtype='float32')
    # objpoints3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32')
    # KK = np.array(internal_calibration, dtype='float32')
    corners2D_gt = np.array(corners2D_gt, dtype='float32')


    # retval,rvector,tvector,inlier=cv.solvePnPRansac(objpoints3D,corners2D_gt,KK,distCoeffs)
    # _,rvector,tvector,inlier=cv.solvePnPRansac(objpoints3D,corners2D_gt,KK,distCoeffs)
    # print(rvector)
    # print(tvector)
    # print(inlier)
    # rmatrix,_ =cv.Rodrigues(rvector)
    # print(rmatrix)
    # theta_x = math.atan2(rmatrix[2,1], rmatrix[2,2])
    # theta_y = math.atan2(-rmatrix[2,0], math.sqrt(rmatrix[2,1] ** 2 + rmatrix[2,2] ** 2))
    # theta_z = math.atan2(rmatrix[1,0], rmatrix[0,0])
    # du_x = (theta_x / math.pi) * 180
    # du_y = (theta_y / math.pi) * 180
    # du_z = (theta_z / math.pi) * 180
    # print(du_x,du_y,du_z)
    # axis = np.float32([[0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)

    # imgpts, _ = cv.projectPoints(axis, rvector, tvector,KK,distCoeffs,)
    # print(imgpts)
    # img = draw(img,(x_pose[0],y_pose[0]),imgpts)
    # 保存照片
    cv.imwrite('predict'+ num + '.jpg',img)
    cv.destroyAllWindows()   
