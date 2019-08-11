# SEU-Advanced-Vision

## 算法

### LINEMOD 数据集制作

数据集制作工具[*ObjectDatasetTools*](https://github.com/seu-labview/ObjectDatasetTools)

### 神经网络

前期选用[*YOLO6D*](https://github.com/seu-labview/singleshot6Dpose)

## 前端

使用*PyQt*完成前端设计

## 接口

使用*librealsense2*调用SR300

### SR300 相机参数

相机内参：*rs-sensor-control*  [教程](https://blog.csdn.net/weixin_39585934/article/details/84147449)

设置参数：*rs2_options* [教程](https://www.greatqq.com/2019/06/intel-realsense-sensors-options/)

## 项目结构

`QT.py`为主文件，实现了整个项目的运作，包含前端显示、调用神经网络、保存文件等。

`Predict.py`为神经网络接口，被`QT.py`调用。

`MeshPly.py`为3D模型接口，被`Predict.py`调用。

使用时，将`LINEMOD`文件夹和`librealsense2.so`置于本项目根目录，`common`和`third-party`文件夹置于`cppcamera`文件夹下。
