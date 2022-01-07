# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import subprocess
import os
import time
import shutil
import argparse
import configparser
import base64
import prettytable
import sys
import uuid
import numpy as np
import cv2
from gevent import monkey
from gevent.pywsgi import WSGIServer
#
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.utils.FileOperationUtil import FileOperationUtil
#

monkey.patch_all()
from flask import Flask, request, jsonify

app = Flask(__name__)


# todo 鸟巢，裸导线（最新的）

# todo 这边有新的任务就会不断地去起，要不要这边判断 GPU 资源是否够 再去决定是否起来？


@app.route('/dete', methods=['post'])
def start_dete():
    """获取检测状态"""

    try:
        model_list = request.form['model_list'].split(',')
        img_path_list = request.form['img_path_list'].split(',')
        post_url = request.form['post_url']
        batch_id = request.form['batch_id']
        #

        print('-'*100)
        print("* model list : {0}, type : {1}".format(model_list, type(model_list)))
        print("* img path list length: {0}, type : {1}".format(len(img_path_list), type(img_path_list)))
        print("* post url : {0}, type : {1}".format(post_url, type(post_url)))
        print("* batch id : {0}, type : {1}".format(batch_id, type(batch_id)))
        print('-'*100)

        assign_save_dir = os.path.join(save_dir, batch_id+ "-" + str(uuid.uuid1()))
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
        img_count = len(img_path_list)
        img_path_txt_path = os.path.join(assign_save_dir, "img_path_to_dete.txt")
        with open(img_path_txt_path, 'w') as txt_file:
            for each_img_path in img_path_list:
                txt_file.write(each_img_path)
                txt_file.write('\n')
        #
        # --------------------------------------------------------------------------------------------------------------
        # 起一个 taizhou_server
        taizhou_server_bug_file = open(os.path.join(log_dir, "taizhou_server_bug" + str(time.time())[:10] + "_allflow.txt"), "w+")
        taizhou_server_std_file = open(os.path.join(log_dir, "taizhou_server_std" + str(time.time())[:10] + "_allflow.txt"), "w+")
        server_cmd_str = r"python3 scripts/server/taizhou/taizhou_server.py --xml_tmp {0} --xml_res {1} --post_url {2} --img_count {3} --mul_progress_num {4} --sign_end_txt_dir {5} --print_process {6} --batch_id {7}".format(
            xml_tmp_dir, xml_res_dir, post_url, img_count, mul_process_num, sign_dir, "True", batch_id)
        pid = subprocess.Popen(server_cmd_str.split(), stdout=taizhou_server_std_file, stderr=taizhou_server_bug_file, shell=False)
        print("* server pid : {0}, cmd str : {1}".format(pid, server_cmd_str))
        # --------------------------------------------------------------------------------------------------------------
        # 起主检测服务
        for i in range(1, mul_process_num + 1):
            each_cmd_str = r"python3 scripts/all_models_flow.py --scriptIndex {0}-{1} --gpuID {2} --model_list {3} --assign_img_dir {4} --ignore_history {5} --del_dete_img {6} --signDir {7} --outputDir {8}".format(
                mul_process_num, i, gpu_id_list[(i - 1) % len(gpu_id_list)], ','.join(model_list), img_path_txt_path, 'True', 'False', sign_dir, assign_save_dir)
            each_bug_file = open(os.path.join(log_dir, "bug{0}_".format(i) + str(time.time())[:10] +  "_allflow.txt"), "w+")
            each_std_file = open(os.path.join(log_dir, "std1{0}_".format(i) + str(time.time())[:10] +  "_allflow.txt"), "w+")
            each_pid = subprocess.Popen(each_cmd_str.split(), stdout=each_std_file, stderr=each_bug_file, shell=False)
            print("pid : {0}".format(each_pid.pid))
            print("* {0}".format(each_cmd_str))
            time.sleep(1)

        return jsonify({"status":"OK"})
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)
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
    parser.add_argument('--save_dir', dest='save_dir', type=str)
    parser.add_argument('--mul_process_num', dest='mul_process_num', type=int, default=2)
    parser.add_argument('--gpu_id_list', dest='gpu_id_list', type=str, default="0")
    #
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    portNum = args.port
    host = args.host
    #
    save_dir = args.save_dir
    mul_process_num = args.mul_process_num
    gpu_id_list = args.gpu_id_list.split(",")
    #
    serv_start()


