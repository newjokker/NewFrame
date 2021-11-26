# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import base64
import prettytable
import sys
import time
import uuid
import numpy as np
import cv2

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')
sys.path.insert(0, lib_path)

import argparse
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


# todo 下面的几个操作都是在维护一个文件

@app.route('/receive_server/post_img/<is_end>', methods=['post'])
def receive_img(is_end):
    """获取检测状态"""

    # fixme 推送的是最后一张的时候，直接加上 txt 记录

    global dete_img_num, img_dir, random_dir_name, batch_size, sign_txt_path

    # a batch img
    if (dete_img_num % batch_size == batch_size-1) and (dete_img_num != -1):
        with open(sign_txt_path, 'a+') as sign_txt_file:
            sign_txt_file.write(random_dir_name + '\n')
        random_dir_name = str(uuid.uuid1())
        random_dir_path = os.path.join(img_dir, random_dir_name)
        os.makedirs(random_dir_path, exist_ok=True)
    elif is_end == 'True':
        with open(sign_txt_path, 'a+') as sign_txt_file:
            sign_txt_file.write(random_dir_name + '\n')

    dete_img_num += 1

    try:
        # img
        upload_file = request.files['image']
        # name
        name = request.files['image'].filename
        # save
        save_dir = os.path.join(img_dir, random_dir_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img_save_path = os.path.join(save_dir, name)
        print(img_save_path)
        # save img
        upload_file.save(img_save_path)

        return jsonify({"status":"OK"})
    except Exception as e:
        return jsonify({"status": "ERROR:{0}".format(e)})

@app.route('/receive_rtsp_server/add_rtsp', methods=['post'])
def add_rtsp():
    """增加视频流"""
    data = request.get_data()
    data = bytes.decode(data)
    data = json.loads(data)
    #
    assign_rtsp = data['rtsp']
    assign_model_list = eval(data['model_list'])
    if assign_rtsp not in rtsp_model_dict:
        rtsp_model_dict[assign_rtsp] = assign_model_list
        jsonify({"status": "success"})
    else:
        jsonify({"status":"error", "message":"rtsp path on dete already"})

@app.route('/receive_rtsp_server/remove_rtsp', methods=['post'])
def remove_rtsp():
    """删除视频流"""
    data = request.get_data()
    data = bytes.decode(data)
    data = json.loads(data)
    #
    assign_rtsp = data['rtsp']
    if assign_rtsp not in rtsp_model_dict:
        jsonify({"status": "error", "message": "rtsp not found"})
    else:
        rtsp_model_dict.pop(assign_rtsp)
        jsonify({"status":"success"})

@app.route('/receive_rtsp_server/change_rtsp_model_list', methods=['post'])
def change_rtsp_model_list():
    """修改视频流对应的模型"""
    data = request.get_data()
    data = bytes.decode(data)
    data = json.loads(data)
    #
    assign_rtsp = data['rtsp']
    assign_model_list = eval(data['model_list'])
    if assign_rtsp not in rtsp_model_dict:
        jsonify({"status": "error", "message": "rtsp not found"})
    else:
        rtsp_model_dict[assign_rtsp] = assign_model_list
        jsonify({"status":"success"})


def serv_start():
    global host, portNum
    http_server = WSGIServer((host, portNum), app)
    http_server.serve_forever()

def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--port', dest='port', type=int, default=3232)
    parser.add_argument('--host', dest='host', type=str, default='0.0.0.0')
    #
    parser.add_argument('--img_dir', dest='img_dir', type=str, default='/usr/input_picture')
    parser.add_argument('--sign_dir', dest='sign_dir', type=str, default='./sign')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=20)
    #
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    portNum = args.port
    host = args.host

    img_dir = args.img_dir
    sign_dir = args.sign_dir
    batch_size = args.batch_size


    # ------------------------------------------------------------------------------------------------------------------
    # todo 直接将图片随机平均放到 n 个文件夹中，设置处理完文件夹中的图片后不删除文件夹

    # ------------------------------------------------------------------------------------------------------------------



    # rtsp 地址和模型列表之间的对应关系




    rtsp_model_dict = {}



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














