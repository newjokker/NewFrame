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

# fixme 小服务的生命周期问题，服务什么时候关闭，由谁去关闭（allflow）当 allflow 检测到已经完成的时候会关闭小服务，一整个检测服务结束了，小服务和核心的 docker 服务就会被关闭

# fixme 怎么才能算检测完毕，就是在 sign 文件夹中增加一个标志文件，启动 docker 的时候就应该输入一个当次要检测图片的最大值，超过这个值 docker 中的一些服务就会被自动关掉，或者只要有图就会自动检测，不存在检测完毕的问题，只有所有图片都已检测完毕

# fixme 张一辰试了一下使用 redis 存储一张图 0.16 s 读取一张图几乎不用时间

# fixme 某张图片检测失败的话会将图片拷贝到一个专门的地址，等待其他图片检测完了之后再去检测


@app.route('/receive_server/post_img', methods=['post'])
def receive_img():
    """获取检测状态"""
    global dete_img_num, img_dir, random_dir_name, batch_size, sign_txt_path

    # a batch img
    if dete_img_num % batch_size == batch_size-1:
        with open(sign_txt_path, 'a+') as sign_txt_file:
            sign_txt_file.write(random_dir_name + '\n')
        random_dir_name = str(uuid.uuid1())
        random_dir_path = os.path.join(img_dir, random_dir_name)
        os.makedirs(random_dir_path, exist_ok=True)

    dete_img_num += 1

    try:
        # img
        upload_file = request.files['image']
        # name
        # name = request.form['filename']
        name = request.files['image'].filename
        # save
        save_dir = os.path.join(img_dir, random_dir_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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
    random_dir_path = os.path.join(img_dir, random_dir_name)
    os.makedirs(random_dir_path, exist_ok=True)

    url = r"http://" + host + ":" +  str(portNum) + "/receive_server"

    print(url)

    # ----------------------------------------------------------------------------------

    serv_start()














