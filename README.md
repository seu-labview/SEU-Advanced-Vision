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

使用时，将`LINEMOD`文件夹置于本项目根目录

##  安装PyTorch对应版本
通过JetPack安装CUDA和CUDNN后记得重新启动

Pytorch 根据对应cuda有不同的版本

本项目可以通过以下语句安装对应的pytorch和torchvision

*sudo pip3 install torch==1.1.0 torchvision==0.3.0  -f https://download.pytorch.org/whl/cu90/stable* 

事实上除了直接编译pytorch源代码外，通过访问对应cuda版本的pytorch下载仓库可以得到官方已经帮你编译好的版本。

例如如果未来某天我们安装了CUDA10就可以将*https://download.pytorch.org/whl/cu90/stable* 中的 *cu90* 换成 *cu100* 即

*https://download.pytorch.org/whl/cu1000/stable*

在那里下载，可以得到一个 stable 文件，这个文件就包含了对应此版本cuda的所有pytorch版本
