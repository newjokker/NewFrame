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

* vim 安装 : apt install vim

* 新增的包，已经对应的版本

    * cv2 : pip install -i https://pypi.douban.com/simple opencv-python
        * 报错：ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
        * 解决：apt-get install libglib2.0-0
    * flask
    * shapely
    * JoUtils 
    * Crypto
        * 报错：ModuleNotFoundError: No module named 'Crypto'
        * 解决：https://cloud.tencent.com/developer/article/1563346
    * matplotlib 
    * pandas 
    * seaborn
    * -------------------------------------------
    * tensorflow-gpu         1.15.0

* 删除测试之后的 ~/.cache 里面的文件夹

* 更改 torch 包中的代码

    * /opt/conda/lib/python3.7/site-packages/torch/nn/init.py 
    
    * /opt/conda/lib/python3.7/site-packages/torch/nn/modules/activation.py
    

### 版本

* 

* environment:v0.0.3 , 没有 tensorflow 只有 torh 的环境(4.13G)

* environment:v0.0.3 , 没有 tensorflow 只有 torh 的环境, 去掉临时文件的缩小版本（3.96G）

* 


