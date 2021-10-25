# -*- coding: utf-8  -*-
# -*- author: jokker -*-


# todo 返回检测进度

# todo 返回检测结果信息，服务读取 xml 解析并返回信息

# todo 开始 | 暂停 检测服务，通过和核心服务之间进行交互（通过删除或者生成一个标志文件实现，每检测一张图片检测一下一个文件是否存在，不存在的话停掉服务, 或者还有什么优雅的方式）

# todo 看  JoTools 中的 for_CSDN 中的 server 怎么写的

# todo 返回

# ----------

# 图片文件夹 | 标志信息文件夹 | xml 存放文件夹 | log 存放文件夹



import os
import prettytable
import os, sys
import time

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')
sys.path.insert(0, lib_path)

import numpy as np
import argparse
import cv2
from gevent import monkey
from gevent.pywsgi import WSGIServer
import datetime

#
from JoTools.txkjRes.deteRes import DeteRes
#

monkey.patch_all()
from flask import Flask, request, jsonify
import threading
import configparser
app = Flask(__name__)



class LogServer(object):

    def __init__(self, img_dir, sign_dir, xml_dir):
        self.img_dir = img_dir
        self.sign_dir = sign_dir
        self.xml_dir = xml_dir
        #
        self.start_time = time.time()
        self.dete_count = 0
        self.img_count = 0

    def get_status(self):
        # 已经检测多少张数据
        # 一共多少张数据
        # 花费了多长时间
        # 取回已经检测的数据

        pass

    def get_dete_res(self):
        pass

    def start_dete(self):
        pass

    def stop_dete(self):
        pass



@app.route('/get_status', methods=['POST'])
def get_status():
    """获取检测状态"""
    return jsonify(log_server.get_status())


@app.route('/get_dete_res', methods=['POST'])
def get_dete_res():
    """获取检测结果"""
    # todo 已经获取的检测结果直接删掉，或者移动到删除文件夹中去
    command_info = request.form['command_info']
    res = parse_command(command_info)
    rsp = {'res': res}
    return jsonify(rsp)

@app.route('/start_dete', methods=['get'])
def start_dete():
    """获取检测结果"""
    # todo 开始检测，创建 sign 文件中检测的标志文件

@app.route('/stop_dete', methods=['get'])
def stop_dete():
    """获取检测结果"""
    # todo 停止检测，删除 sign 文件中检测的标志文件


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
    parser.add_argument('--xml_dir', dest='xml_dir', type=str, default='/')
    parser.add_argument('--sign_dir', dest='sign_dir', type=str, default='/')
    #
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    portNum = args.port
    host = args.host

    log_server = LogServer(args.img_dir, args.xml_dir, args.sign_dir)

    url = r"http://" + host + ":" +  str(portNum) + "/record_find"
    print(url)

    # ----------------------------------------------------------------------------------

    a = RecordFind()
    serv_start()














