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
    * tensorflow-gpu         1.15.0 : pip install -i https://pypi.douban.com/simple tensorflow-gpu==1.15.0
    * sklearn
    * 

* 删除测试之后的 ~/.cache 里面的文件夹

* 更改 torch 包中的代码

    * /opt/conda/lib/python3.7/site-packages/torch/nn/init.py 
    
    * /opt/conda/lib/python3.7/site-packages/torch/nn/modules/activation.py

* lib 中要修改的部分（与 2021.11.5时的版本）

    * JoTools 需要支持 get_img_by_dete_obj_new , 也就是直接支持插入 data 进行 crop 的版本
    
    * yolov5_libs 版本，yolov5_libs 里面有很多需要修改的地方，现在编译为 so 了看不到了，有时间重新写一下
    
    * lib/r2cnnPytorch_libs/maskrcnn_benchmark 中缺少一个文件，需要拷贝过去，这个是在运行环境上编译而成的：_C.cpython-37m-x86_64-linux-gnu.so
    
    * xjDeeplabDetection 中相关的 print 进行删除

* model 进行加密

    * fzc step 1 要使用 py35 版本的，因为那个版本的 torch 版本和 docker 中的 torch 版本一致
    
* 模型要支持多进程
    * /home/ldq/v0.0.1/lib/detect_libs/abstractBase.py 中修改代码
    * 将模型解压到随机文件夹中，这样就能同时启动两套模型了 
    `self.cachePath = os.path.join(os.path.expanduser('~'), '.cache')
    self.cachePath = os.path.join(self.cachePath, str(uuid.uuid1()))
    `
    
    

### 版本

* 

* environment:v0.0.3 , 没有 tensorflow 只有 torh 的环境(4.13G)

* environment:v0.0.4 , 没有 tensorflow 只有 torh 的环境, 去掉临时文件的缩小版本（3.96G）

* 


