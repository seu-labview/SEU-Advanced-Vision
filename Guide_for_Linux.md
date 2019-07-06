# Guide for Linux

Since we haven't decided which OS to choose, I had to have tried both (Win10 and Ubuntu, of course).
This document is for environment setting in Linux.

## Install Ubuntu 16.04LTS

RTFM and STFW, and U can do it. Attention, at least 30GB space of hard disk is needed.

**PLZ back up your data** if you are not familiar with OS installation.

If you are in special regions, **PLZ change the downloading source**.
I recommend using the built-in software *Software and Updates*(*软件与更新*）to change the source.

## Install GPU driver

Nvidia

## Install CUDA

PLZ PLZ install CUDA via *JetPack*!
It's quite safe and sound!
Click [here](https://developer.nvidia.com/embedded/downloads) to find and download the latest release of *JetPack*!

Anyway you'll need a Nvidia&reg; developer account.

We only need to install CUDA and openCV, so make sure that you have and only have ticked CUDA and openCV.

## Install TensorFlow

This step is pretty easy, but also **PLZ change download source of pip** if you are in special regions.

Official installation guide is [here](https://tensorflow.google.cn/install/source).
I strongely recommend choosing the stable one with GPU.

You need to compile with [Bazel 10](https://docs.bazel.build/versions/master/install-ubuntu.html).

PLZ **don't** install via pip.

## Install PoseCNN

***
TO BE CONTINUED
