# 接口文档


### 风火轮接口文档

#### 起服务

* docker run --gpus 'device=0' -v /home/ldq/input_dir:/usr/input_picture -v /home/ldq/sign:/v0.0.1/sign -v /home/ldq/output_dir:/usr/output_dir  -it wuhan_ft:v0.0.0 /bin/bash 

* python3 hot_wheel_log_server.py --host 127.0.0.1 --port 8802 --img_dir /home/ldq/input_dir --xml_dir /home/ldq/output_dir/save_res

* python3 hot_wheel_receive_server.py --host 127.0.0.1 --port 8803 --img_dir /home/ldq/input_dir  --sign_dir /home/ldq/sign --batch_size 20

* python3 hot_wheel_post_server.py --host 127.0.0.1 --port 8008 --xml_dir /home/ldq/output_dir/save_res --post_mode 0  

#### 访问检测信息

* /log_server/get_status : info = {'dete_img_num':-1, 'use_time':-1, 'total_img_num':-1}

* /log_server/get_dete_res : each_dete_res.get_fzc_format()  {'xml_name': 'each_dete_res'} 【按照目前的方案，不需要】

#### 推送地址

* host:port/receive_server/post_img/<is_end>  【hot_wheel_receive_server.py】
    * <is_end> 
        * type : str
        * valuse 'True' | 'False'
        * info : 推送的最后一张图片参数设置为 True 否者设置为 False 
    * 推送的图片格式
        * data={'filename': each_img_name}
        * files={'image': open(each_img_path, 'rb')}



