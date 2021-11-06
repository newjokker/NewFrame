# 缩小体积


### 步骤

* docker 中下载 ubuntu16.04 镜像 , docker pull ubuntu:16.04

* docker 中安装最基础的工具
    * vim
    * make
    * gcc

* docker 中安装 python, https://blog.csdn.net/weixin_43901865/article/details/115345397



### 需要修改的 torch 中的内容

* docker pull pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

* 更新后的文件放在当前文件所在的文件夹中

* 新增的包，已经对应的版本

    * tensorflow-gpu         1.15.0
    * 
   

* /opt/conda/lib/python3.7/site-packages/torch/nn/init.py 

* /opt/conda/lib/python3.7/site-packages/torch/nn/modules/activation.py


# todo 环境写为 一个 docker file 直接打包



### 细节

* 只有环境没有模型和代码的版本 fangtian:v0.4.1.3

* 



