# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import prettytable
import os, sys
import time

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


class LogServer(object):

    def __init__(self, xml_dir, sign_dir, img_count=-1):
        self.sign_dir = sign_dir
        self.xml_dir = xml_dir
        self.xml_dir_return = os.path.join(self.sign_dir, "returned_xml")            # 已经返回结果的 xml 路径
        os.makedirs(self.xml_dir_return, exist_ok=True)
        #
        self.start_time = time.time()
        self.stop_time = None
        self.dete_count = 0
        self.img_count = img_count
        #
        self.buffer_xml_path_list = []      # 还未返回结果的图片 xml

        print("xml dir : {0}".format(self.xml_dir))
        print("sign dir : {0}".format(self.sign_dir))

    def get_status(self):
        info = {'dete_img_num':-1, 'use_time':-1, 'total_img_num':-1}
        xml_path_list = list(FileOperationUtil.re_all_file(self.xml_dir, endswitch=['.xml']))
        dete_count = len(xml_path_list) + self.dete_count
        # fixme 这边去定义结束时间并不正确，要修改一下
        if dete_count == self.img_count and self.stop_time is None:
            self.stop_time = time.time()
        #
        if self.stop_time:
            use_time = self.stop_time - self.start_time
        else:
            use_time = time.time() - self.start_time
        #
        info['dete_img_num'] = dete_count
        info['use_time'] = use_time
        info['total_img_num'] = self.img_count
        #
        return info

    def get_dete_res(self, assign_img_count=-1):
        # get xml path

        # todo 返回指定个数的文件结果，默认值为 -1 是返回 buffer 中所有的结果

        dete_res_dict = {}
        xml_path_list = (FileOperationUtil.re_all_file(self.xml_dir, endswitch=['.xml']))
        for each_xml_path in xml_path_list:
            if each_xml_path not in self.buffer_xml_path_list:
                self.buffer_xml_path_list.append(each_xml_path)
        # merge dete res
        for each_xml_path in self.buffer_xml_path_list:
            each_dete_res = DeteRes(each_xml_path)
            xml_name = FileOperationUtil.bang_path(each_xml_path)[1]
            dete_res_dict[xml_name] = each_dete_res.get_fzc_format()
        # empty buffer xml
        self.dete_count += len(self.buffer_xml_path_list)
        FileOperationUtil.move_file_to_folder(self.buffer_xml_path_list, self.xml_dir_return, is_clicp=True)
        self.buffer_xml_path_list = []
        return dete_res_dict

    def get_his_dete_res(self, img_name):
        """获取历史检测结果"""
        pass

    def start_dete(self):
        """没有之间的文件，就不进行检测"""
        start_sign_txt_path = os.path.join(self.sign_dir, 'start_dete.txt')
        sign_txt = open(start_sign_txt_path, 'w')
        sign_txt.close()

    def stop_dete(self):
        start_sign_txt_path = os.path.join(self.sign_dir, 'start_dete')
        if os.path.exists(start_sign_txt_path):
            os.remove(start_sign_txt_path)


@app.route('/log_server/get_status', methods=['get'])
def get_status():
    """获取检测状态"""
    return jsonify(log_server.get_status())

@app.route('/log_server/get_dete_res', methods=['POST'])
def get_dete_res():
    """获取检测结果"""
    # command_info = request.form['command_info']
    # res = parse_command(command_info)
    return jsonify(log_server.get_dete_res())

@app.route('/log_server/start_dete', methods=['get'])
def start_dete():
    """获取检测结果"""
    log_server.start_dete()
    return jsonify({"status":"OK"})

@app.route('/log_server/stop_dete', methods=['get'])
def stop_dete():
    """获取检测结果"""
    log_server.stop_dete()
    return jsonify({"status":"OK"})

def serv_start():
    global host, portNum
    http_server = WSGIServer((host, portNum), app)
    http_server.serve_forever()

def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--port', dest='port', type=int, default=3232)
    parser.add_argument('--host', dest='host', type=str, default='192.168.3.74')   # 这边要是写 127 的话只能在本服务器访问了，要改为本机的地址
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

    log_server = LogServer(args.xml_dir, args.sign_dir, 1000000)

    url = r"http://" + host + ":" +  str(portNum) + "/log_server"

    print(url)

    # ----------------------------------------------------------------------------------

    serv_start()














