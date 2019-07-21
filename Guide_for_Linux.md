# Guide for Linux

Since we haven't decided which OS to choose, I had to have tried both (Win10 and Ubuntu, of course).
This document is for environment setting in Linux.

[Here](https://github.com/Kaju-Bubanja/PoseCNN) is someone else's guide. 

## Install Ubuntu 16.04LTS

RTFM and STFW, and U can do it. Attention, at least 30GB space of hard disk is needed.

**PLZ back up your data** if you are not familiar with OS installation.

If you are in special regions, **PLZ change the downloading source**.
I recommend using the built-in software *Software and Updates*(*软件与更新*）to change the source.

## Install GPU driver

You can find this in the built-in software *Software and Updates*(*软件与更新*）.

## Install GCC 4.8

Normally you need to remove the original gcc, and install v4.8.

```Bash
sudo apt remove gcc
sudo apt install gcc-4.8
```

## Install CUDA 9.0 and cudNN 7.1.4

PLZ PLZ install CUDA via *JetPack 3.3*!
It's quite safe and sound!
Click [here](https://developer.nvidia.com/embedded/downloads) to find and download *JetPack 3.3*!

Anyway you'll need a Nvidia&reg; developer account.

We only need to install CUDA and openCV, so make sure that you have and only have ticked CUDA and openCV.

As for cuDNN, I strongly recommand binary install package.

## Install TensorFlow r1.8

This step is pretty easy, but also **PLZ change download source of pip** if you are in special regions.

Official installation guide is [here](https://tensorflow.google.cn/install/source).
I strongely recommend choosing the stable one with GPU.

**We should use version r1.8**.

You need to compile with [Bazel 10](https://docs.bazel.build/versions/master/install-ubuntu.html).

While setting configuration, I recommand to choose 'N' for all **except cuda**.

https://www.jianshu.com/p/a64bd41c585e

## Install PoseCNN

### Pangolin

To install [*Pangolin*](https://github.com/uoip/pangolin), you need to install glew first:`sudo apt-get install libglew-dev`

Pangolin install guide can be seen at the official repo.

### nanoflann

Download the archive and unzip it at somewhere.

```Bash
#How to install a cmake project like nanoflann
cd nanoflann-<version>
mkdir build
cmake ..
make
make install
```

***
TO BE CONTINUED
