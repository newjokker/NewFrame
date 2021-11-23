
# python 3.6.9 | torch 1.8.1 | torchvision 0.9.1
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# -y：yes，在命令行交互提示中，直接输入 yes
RUN apt-get update && apt-get install vim -y && apt-get install libglib2.0-0 -y && apt install libgl1-mesa-glx -y && apt-get install gcc -y

# create dictory and se as work dictory
RUN mkdir /v0.0.1 && mkdir /v0.0.1/docker_tmp
WORKDIR /v0.0.1
COPY ./data/JoUtil-0.2.12.9-py3-none-any.whl /v0.0.1/docker_tmp

# install python package
RUN pip install -i https://pypi.douban.com/simple \
    flask\
    shapely\
    pycryptodome\
    Cython\
    matplotlib\
    pandas\
    seaborn\
    sklearn\
    yacs\
    tensorflow-gpu==1.15.0\
    opencv-python\
    /v0.0.1/docker_tmp/JoUtil-0.2.12.9-py3-none-any.whl

# remove temp file
RUN rm -r /v0.0.1/docker_tmp

# copy all code to docker
COPY ./data/v0.0.1 /v0.0.1


# model 最后一起加密


