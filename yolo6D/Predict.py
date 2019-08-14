import numpy as np
from cv2 import cv2
import os
import math
import os
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
import threading
import queue

from yolo6D.darknet import Darknet as dn
import yolo6D.dataset
from yolo6D.utils import get_camera_intrinsic, do_detect, get_3D_corners
from yolo6D.MeshPly import MeshPly


def draw_predict(bss, strss, img, num):
    '''绘制预测框'''
    j = 0
    for bs in bss:
        width = 640
        height = 480
        x = [0 for i in range(9)]
        y = [0 for i in range(9)]
        a = np.arange(18)
        corners2D_gt = a.reshape(9, 2)

        for i in range(9):
            x[i] = int(bs[i][0] * width)
            y[i] = int(bs[i][1] * height)
            cv2.circle(img, (x[i], y[i]), 1, (255, 0, 255), -1)
            string = str(i)
            cv2.putText(
                img, string, (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
            corners2D_gt[i, 0] = x[i]
            corners2D_gt[i, 1] = y[i]

        cv2.line(img, (x[1], y[1]), (x[2], y[2]), (255, 255, 0), 2)
        cv2.line(img, (x[2], y[2]), (x[4], y[4]), (255, 255, 0), 2)
        cv2.line(img, (x[3], y[3]), (x[4], y[4]), (255, 255, 0), 2)
        cv2.line(img, (x[1], y[1]), (x[3], y[3]), (255, 255, 0), 2)

        cv2.line(img, (x[1], y[1]), (x[5], y[5]), (255, 255, 0), 2)
        cv2.line(img, (x[2], y[2]), (x[6], y[6]), (255, 255, 0), 2)
        cv2.line(img, (x[3], y[3]), (x[7], y[7]), (255, 255, 0), 2)
        cv2.line(img, (x[4], y[4]), (x[8], y[8]), (255, 255, 0), 2)

        cv2.line(img, (x[5], y[5]), (x[6], y[6]), (255, 255, 0), 2)
        cv2.line(img, (x[5], y[5]), (x[7], y[7]), (255, 255, 0), 2)
        cv2.line(img, (x[6], y[6]), (x[8], y[8]), (255, 255, 0), 2)
        cv2.line(img, (x[7], y[7]), (x[8], y[8]), (255, 255, 0), 2)

        # 输出物品名称和识别率
        text = strss[j][0] + ' ' + strss[j][1]
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1)
        if x[1] + size[0][0] <= width and y[1] - size[0][1] > 0:
            tx = x[1]
            ty = y[1]
        else:
            tx = x[8]
            ty = y[8]
        cv2.rectangle(img, (tx - 2, ty + 2), (tx + 2 +
                                             size[0][0], ty - 2 - size[0][1]), (255, 255, 0), cv2.FILLED)
        cv2.putText(img, text, (tx, ty),
                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0))

        j += 1

    # 保存照片
    cv2.imwrite('JPEGImages/predict' + str(num) + '.jpg', img)


def makedirs(path):
    '''Create new directory'''
    if not os.path.exists(path):
        os.makedirs(path)


def detect(name, model, image_path):
    '''
    调用神经网络检测
    输入：图片位置
    返回：9个二维点坐标和组成的2*10数组
    '''
    # Parameters
    seed = int(time.time())
    gpus = '0'     # Specify which gpus to use
    torch.manual_seed(seed)
    use_cuda = False
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    img = cv2.imread(image_path, 1)
    return do_detect(model, img, 0.1, 0.4, 0)


def predict(name, model, num):
    '''
    ！注意：需将权重文件放在 项目根目录中 weigths 下
    输入: 物体名称，模型，图片编号
    返回：长度为20的数组，前18为坐标，后接物品名和识别率
    '''
    img_name = 'JPEGImages/' + str(num) + '.jpg'

    boxes = detect(str(name), model, img_name)
    best_conf_est = -1
    for j in range(len(boxes)):
        if (boxes[j][18] > best_conf_est):
            box_pr = boxes[j]
            best_conf_est = boxes[j][18]

    strs = []
    strs.append(name)
    strs.append(str(int(best_conf_est.numpy() * 100)) + '%')

    return np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32'), strs


class predict_thread(threading.Thread):
    '''
    q：状态数组，name：物品名称，num： 编号队列
    '''

    def __init__(self, q, name, model, numq, strs):
        threading.Thread.__init__(self)
        self.q = q
        self.name = name
        self.model = model
        self.numq = numq
        self.strs = strs  # 物品名称，识别率（字符串）

    def run(self):
        num = self.numq.get()
        bs, strs = predict(self.name, self.model, num)
        self.q.put(bs)
        self.strs.put(strs)
        print('\t\033[0;32m%s预测完毕\033[0m' % self.name)
