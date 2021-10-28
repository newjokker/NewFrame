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

# fixme 小服务的生命周期问题，服务生命时候关闭，由谁去关闭（allflow）当 allflow 检测到已经完成的时候会关闭小服务

@app.route('/receive_server/post_img', methods=['post'])
def receive_img():
    """获取检测状态"""
    global dete_img_num, img_dir, random_dir_name, batch_size

    # a batch img
    if dete_img_num % batch_size == 0:
        random_dir_name = str(uuid.uuid1())
        random_dir_path = os.path.join(img_dir, random_dir_name)
        os.makedirs(random_dir_path, exist_ok=True)
        # todo 在 sign 文件夹中记录需要检测数据 txt 中的一行记录
        with open(sign_txt_path, 'a+') as sign_txt_file:
            sign_txt_file.write(random_dir_name + '\n')

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
    parser.add_argument('--img_dir', dest='img_dir', type=str, default='/temp')
    parser.add_argument('--sign_dir', dest='sign_dir', type=str, default='/temp')
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

    dete_img_num = 0

    sign_txt_path = os.path.join(sign_dir, 'img_dir_to_dete.txt')
    if os.path.exists(sign_txt_path):
        os.remove(sign_txt_path)

    random_dir_name = str(uuid.uuid1())

    url = r"http://" + host + ":" +  str(portNum) + "/receive_server"

    print(url)

    # ----------------------------------------------------------------------------------

    serv_start()














