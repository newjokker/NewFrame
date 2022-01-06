# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import subprocess
import os
import time
import shutil
import argparse
import configparser
from JoTools.utils.FileOperationUtil import FileOperationUtil
from JoTools.txkjRes.deteRes import DeteRes
import base64
import prettytable
import sys
import uuid
import numpy as np
import cv2
from gevent import monkey
from gevent.pywsgi import WSGIServer
import datetime
#
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.utils.FileOperationUtil import FileOperationUtil
#

monkey.patch_all()
from flask import Flask, request, jsonify

app = Flask(__name__)


# todo 鸟巢，裸导线（最新的）


@app.route('/dete', methods=['post'])
def start_dete():
    """获取检测状态"""

    try:

        model_list = request.form['model_list']
        img_path_list = request.form['img_path_list']
        post_usr = request.form['post_url']
        #
        assign_save_dir = os.path.join(save_dir, str(uuid.uuid1()))
        os.makedirs(assign_save_dir, exist_ok=True)
        xml_tmp_dir = os.path.join(assign_save_dir, "xml_tmp")
        os.makedirs(xml_tmp_dir, exist_ok=True)
        xml_res_dir = os.path.join(assign_save_dir, "xml_res")
        os.makedirs(xml_res_dir, exist_ok=True)
        sign_dir = os.path.join(assign_save_dir, "sign")
        os.makedirs(sign_dir, exist_ok=True)
        log_dir = os.path.join(assign_save_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        # save img path list to txt
        img_path_txt_path = os.path.join(assign_save_dir, "img_path_to_dete.txt")
        with open(img_path_txt_path, 'w') as txt_file:
            for each_img_path in img_path_list:
                txt_file.write(each_img_path)
                txt_file.write('\n')
        #

        # --------------------------------------------------------------------------------------------------------------
        # todo 起一个 taizhou_server

        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # todo 起主检测服务
        for i in range(1, mul_process_num + 1):
            each_cmd_str = r"python3 scripts/all_models_flow.py --scriptIndex {0}-{1} --gpuID {2} --model_list {3} --assign_img_dir {4} --ignore_history {5} --del_dete_img {6} --signDir {7}".format(
                mul_process_num, i, gpu_id_list[(i - 1) % gpu_num], ','.join(model_list), img_path_txt_path, 'True', 'False', sign_dir)
            each_bug_file = open(os.path.join(log_dir, "bug{0}_".format(i) + str(time.time())[:10] +  "_allflow.txt"), "w+")
            each_std_file = open(os.path.join(log_dir, "std1{0}_".format(i) + str(time.time())[:10] +  "_allflow.txt"), "w+")
            each_pid = subprocess.Popen(each_cmd_str.split(), stdout=each_std_file, stderr=each_bug_file, shell=False)
            print("pid : {0}".format(each_pid.pid))
            print("* {0}".format(each_cmd_str))
            time.sleep(1)

        return jsonify({"status":"OK"})
    except Exception as e:
        return jsonify({"status": "ERROR:{0}".format(e)})



def serv_start():
    global host, portNum
    http_server = WSGIServer((host, portNum), app)
    http_server.serve_forever()

def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--port', dest='port', type=int, default=3232)
    parser.add_argument('--host', dest='host', type=str, default='0.0.0.0')
    #
    parser.add_argument('--model_list', dest='model_list', type=str)
    parser.add_argument('--save_dir', dest='save_dir', type=str)
    parser.add_argument('--mul_process_num', dest='mul_process_num', type=int, default=2)
    #
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    portNum = args.port
    host = args.host

    save_dir = args.save_dir
    model_list = args.model_list
    mul_process_num = args.mul_process_num


    dete_img_num = -1

    sign_txt_path = os.path.join(sign_dir, 'img_dir_to_dete.txt')
    if os.path.exists(sign_txt_path):
        os.remove(sign_txt_path)

    random_dir_name = str(uuid.uuid1())
    random_dir_path = os.path.join(img_dir, random_dir_name)
    os.makedirs(random_dir_path, exist_ok=True)

    url = r"http://" + host + ":" +  str(portNum) + "/receive_server"

    print(url)

    # ----------------------------------------------------------------------------------

    serv_start()




# # todo 有一个服务等待着新的任务
#
# # todo 每一个新的任务分配一个随机命名文件夹，xml 和 sign_dir 全部新建在里面
#
# # todo 解析一个新的任务，将要检测的文件路径列表，按照比赛的样式，当做固定文件夹传入即可
#
# # todo 对于同名的文件如何处理，可以将 xml 的名字改为随机的，要知道图片的名字，读取 dete_res.file_name 属性
#
# # fixme 对比的是 经过 ft_server 处理过，的那些 xml ，
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
#     parser.add_argument('--mul_process_num', dest='mul_process_num', type=int, default=2)
#     parser.add_argument('--gpu_id_list', dest='gpu_id_list', type=str, default='0')
#     parser.add_argument('--model_list', dest='model_list', type=str, default='')
#     parser.add_argument('--model_types', dest='model_types', type=str, default='01,02,03')
#     #
#     args = parser.parse_args()
#     return args
#
#
#
#
# if __name__ == "__main__":
#
#
#     # ------------------------------------------------------------------------------------------------------------------
#     args = parse_args()
#     mul_process_num = args.mul_process_num
#     model_list_str = args.model_list
#     model_list = model_list_str.strip().split(',')
#     gpu_id_list_str = args.gpu_id_list
#     gpu_id_list = gpu_id_list_str.strip().split(',')
#     gpu_num = len(gpu_id_list)
#     model_types = args.model_types.strip().split(',')
#
#     # if no model to dete : exit
#     if len(model_list) == 0:
#         print("* no model to dete")
#         exit()
#     else:
#         print("* dete model include : {0}".format(', '.join(model_list)))
#
#     # empty history
#     if os.path.exists(res_xml_tmp):
#         for each_file_path in FileOperationUtil.re_all_file(res_xml_tmp):
#             os.remove(each_file_path)
#
#     # start dete servre
#     for i in range(1, mul_process_num + 1):
#         each_cmd_str = r"python3 scripts/all_models_flow.py --scriptIndex {0}-{1} --gpuID {2} --model_list {3} --assign_img_dir {4} --ignore_history {5} --del_dete_img {6}".format(
#             mul_process_num, i, gpu_id_list[(i-1)%gpu_num], ','.join(model_list), r'/usr/input_picture', 'True', 'False')
#
#         each_bug_file = open(os.path.join(log_dir, "bug{0}_".format(i) + time_str + obj_name + ".txt"), "w+")
#         each_std_file = open(os.path.join(log_dir, "std1{0}_".format(i) + time_str + obj_name + ".txt"), "w+")
#
#         each_pid = subprocess.Popen(each_cmd_str.split(), stdout=each_std_file, stderr=each_bug_file, shell=False)
#         print("pid : {0}".format(each_pid.pid))
#         print("* {0}".format(each_cmd_str))
#         time.sleep(1)
#
#     # start fangtian server
#     start_ft_server(res_xml_tmp, img_dir, res_dir, sign_dir, mul_process_num)
#
#     # 等待
#     while True:
#         time.sleep(60)
#
#     # ------------------------------------------------------------------------------------------------------------------
