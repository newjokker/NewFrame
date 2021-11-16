
# docker pull pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

# -y：yes，在命令行交互提示中，直接输入 yes
RUN apt-get update && apt-get install vim -y && apt-get install libglib2.0-0 -y && apt install libgl1-mesa-glx -y

# create dictory and se as work dictory
RUN mkdir /v0.0.1 && mkdir /v0.0.1/docker_tmp
WORKDIR /v0.0.1
COPY ./data/JoUtil-0.2.12.9-py3-none-any.whl /v0.0.1/docker_tmp

# install python package
RUN pip install -i https://pypi.douban.com/simple \
    flask\
    shapely\
    pycryptodome\
    matplotlib\
    pandas\
    seaborn\
    sklearn\
    yacs\
    tensorflow-gpu==1.15.0\
    opencv-python\
    /v0.0.1/docker_tmp/JoUtil-0.2.12.9-py3-none-any.whl

# update files in toech environment
RUN rm  /opt/conda/lib/python3.7/site-packages/torch/nn/init.py \
        /opt/conda/lib/python3.7/site-packages/torch/nn/modules/activation.py

COPY ./data/init.py /opt/conda/lib/python3.7/site-packages/torch/nn
COPY ./data/activation.py /opt/conda/lib/python3.7/site-packages/torch/nn/modules

# remove temp file
RUN rm -r /v0.0.1/docker_tmp

# copy all code to docker
COPY ./data/v0.0.1 /v0.0.1

#RUN python3 script/all_models_flow.py

# start container and run the app
#docker run --gpus 'device=0' -v /home/ldq/input_dir:/usr/input_picture -v /home/ldq/output_dir:/usr/output_dir -v /home/ldq/json_dir:/usr/input_picture_attach -v /home/ldq/logs:/v0.0.1/logs -it wuhan_ft:v0.0.2 /bin/bash
#python3 script/allflow_wuhan.py

# run startServer.sh at the end
#CMD ["/v0.0.1/startServe.sh"]


