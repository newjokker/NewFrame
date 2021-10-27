# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import prettytable
import os, sys
import time
import uuid

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

    # todo 文件到达多少张就会放到一个文件夹中
    # todo 每次检测一个文件夹中的数据，一次性推送给 java

    global dete_img_num, img_dir, random_dir_name

    dete_img_num += 1

    # a batch img
    if dete_img_num % 200 == 0:
        random_dir_name = str(uuid.uuid1())
        # todo 在 sign 文件夹中记录需要检测数据 txt 中的一行记录

    try:
        # img
        base64_code = request.files['image'].stream.read()
        img_array = np.fromstring(base64_code, np.uint8)
        im = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
        # name
        # name = request.form['filename']
        name = request.files['image'].filename
        # save
        save_dir = os.path.join(img_dir, random_dir_name)
        img_save_path = os.path.join(save_dir, name + '.jpg')
        cv2.imwrite(img_save_path, im)
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
    parser.add_argument('--host', dest='host', type=str, default='192.168.3.155')   # 这边要是写 127 的话只能在本服务器访问了，要改为本机的地址
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

    dete_img_num = 0

    random_dir_name = str(uuid.uuid1())


    url = r"http://" + host + ":" +  str(portNum) + "/receive_server"

    print(url)

    # ----------------------------------------------------------------------------------

    serv_start()














