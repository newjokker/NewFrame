# 说明



* 要执行的命令


* -itd : 
    * -d: 后台运行容器，并返回容器ID
    * -i: 以交互模式运行容器，通常与 -t 同时使用
    * -t: 为容器重新分配一个伪输入终端，通常与 -i 同时使用
    * --rm : 退出之后删除容器

* docker run --gpus device=0 -v /home/ldq/input_dir:/usr/input_picture -v /home/ldq/ou_dir:/usr/output_dir -itd wuhan_ft:v2.1.3 /v0.0.1/start_server.sh