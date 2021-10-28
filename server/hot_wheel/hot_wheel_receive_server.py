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


# todo sign 文件夹是不是可以传输一些什么重要的信息在里面，比如 里面有一个文件记录了下一个要检测的文件夹的名字，随机的 uuid


@app.route('/receive_server/post_img', methods=['post'])
def receive_img():
    """获取检测状态"""

    global dete_img_num, img_dir, random_dir_name

    # a batch img
    if dete_img_num % 20 == 0:
        random_dir_name = str(uuid.uuid1())
        random_dir_path = os.path.join(img_dir, random_dir_name)
        os.makedirs(random_dir_path, exist_ok=True)
        # todo 在 sign 文件夹中记录需要检测数据 txt 中的一行记录

    dete_img_num += 1

    try:
        # img
        upload_file = request.files['image']
        # name
        # name = request.form['filename']
        name = request.files['image'].filename
        # save
        save_dir = os.path.join(img_dir, random_dir_name)
        img_save_path = os.path.join(save_dir, name)
        # save img
        upload_file.save(img_save_path)

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
    parser.add_argument('--img_dir', dest='img_dir', type=str, default='/')
    parser.add_argument('--sign_dir', dest='sign_dir', type=str, default='/')
    #
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    portNum = args.port
    host = args.host

    img_dir = args.img_dir
    sign_dir = args.sign_dir

    dete_img_num = 0

    random_dir_name = str(uuid.uuid1())


    url = r"http://" + host + ":" +  str(portNum) + "/receive_server"

    print(url)

    # ----------------------------------------------------------------------------------

    serv_start()














