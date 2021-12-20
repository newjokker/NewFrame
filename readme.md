# 说明


### 每张图片特殊处理

* 输入时，给每张图片定制一个命令文件，要是没有指定的文件就启动默认的是被，有的话就按照文件中的内容进行启动



* 要执行的命令


* -itd : 
    * -d: 后台运行容器，并返回容器ID
    * -i: 以交互模式运行容器，通常与 -t 同时使用
    * -t: 为容器重新分配一个伪输入终端，通常与 -i 同时使用
    * --rm : 退出之后删除容器

* docker run --gpus device=0 -v /home/ldq/input_dir:/usr/input_picture -v /home/ldq/ou_dir:/usr/output_dir -itd wuhan_ft:v2.1.3 /v0.0.1/start_server.sh

* 使用指定参数
* docker run --gpus device=0 -v /home/ldq/input_dir:/usr/input_picture -v /home/ldq/output_test:/usr/output_dir -e mul_process_num=2 -e gpu_id_list=0 -e model_list=nc,jyzZB,fzc,fzcRust,kkxTC,kkxQuiting,kkxClearance -it  test:v2 /v0.0.1/start_server.sh

* 武汉那边指定的参数，测试的时候不能在后面加 /bin/bash 否者会启动不了，直接 exit 
* docker run --gpus device=0 -v /etc/localtime:/etc/localtime:ro -v /home/ldq/input_dir:/usr/input_picture:ro -v /home/ldq/output_test:/usr/output_dir -v /home/ldq/json_dir:/usr/input_picture_attach -e MODEL_TYPES=00,01 -e NMS=0.6 -e SCORE=0.3 -d wuhan_ft:v2.4.2



