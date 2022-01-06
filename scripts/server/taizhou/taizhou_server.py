# -*- coding: utf-8  -*-
# -*- author: jokker -*-

"""
* 定时扫 xml 文件夹，推送检测结果
"""

import os
import time
import shutil
import csv
import argparse
from JoTools.utils.FileOperationUtil import FileOperationUtil
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.utils.CsvUtil import CsvUtil
from JoTools.utils.JsonUtil import JsonUtil
import requests
from flask import Blueprint, request, jsonify
import json

# todo 增加了推送检测结果到指定位置的功能

# todo 会提前告诉服务要检测多少张图片，检测完了就会直接关掉这个服务（超时也会关闭这个服务）

# todo 对于每一次检测请求，会新建一个随机的文件夹，xml_tmp，xml_res，sign_end_txt_dir 都放在这边

# todo 图像先进行一次解析，不能出现一样的文件路径


class TZserver(object):

    def __init__(self, xml_tmp, xml_res, post_url, img_count, sign_end_txt_dir, mul_progress_num, print_proecss="True"):
        self.xml_tmp = xml_tmp
        self.xml_res = xml_res
        self.post_url = post_url
        self.img_count = img_count
        self.img_index = 0
        #
        self.sign_end_txt_dir = sign_end_txt_dir                # 用于监测模型是不是已经运行成功了，状态文件夹
        #
        self.print_process = eval(print_proecss)
        #
        self.mul_progress_num = mul_progress_num
        #
        self.start_time = time.time()

    def if_end(self):
        """根据 sign 文件夹中的信息，判断是否已经结束 | 根据已检测的图片和图片的总数是否相等"""

        if self.img_index >= self.img_count:
            return True

        for i in range(1, self.mul_progress_num+1):
            each_txt_path = os.path.join(self.sign_end_txt_dir, "{0}.txt".format(i))
            if not os.path.exists(each_txt_path):
                return False
        return True

    def post_res(self, each_xml_path, headers=None):

        dete_res = DeteRes(each_xml_path)

        data = {
            "file_name": "",
            "start_time": "",
            "end_time":"",
            "width": -1,
            "height": -1,
            "count": -1,
            "alarms": []
        }
        # alarms
        alarms, id = [], 0
        for each_dete_obj in dete_res:
            id += 1
            alarms.append([id, each_dete_obj.x1, each_dete_obj.y1, each_dete_obj.x2, each_dete_obj.y2, each_dete_obj.tag, each_dete_obj.conf])
        # time
        des = dete_res.des.split("-")
        start_time, end_time = float(des[0]), float(des[1])
        #
        data["file_name"] = dete_res.file_name
        data["start_time"] = start_time
        data["end_time"] = end_time
        data["width"] = dete_res.width
        data["height"] = dete_res.height
        data["count"] = id
        data["alarms"] = alarms
        #
        if headers is None:
            headers = {'Connection': 'close'}
        try:
            response_data = requests.post(url=self.post_url,  data=json.dumps(data), headers=headers)
            return response_data
        except Exception as e:
            print("----error----")
            print(e)

    def start_monitor(self):
        """开始监听"""
        #
        while True:
            if self.if_end():
                return

            # ----------------------------------------------------------------------------------------------------------
            xml_path_list = list(FileOperationUtil.re_all_file(self.xml_tmp_dir, endswitch=['.xml']))
            #
            for each_xml_path in xml_path_list:
                try:
                    self.post_res(each_xml_path)
                    #
                    if os.path.exists(each_xml_path):
                        new_xml_path = os.path.join(self.xml_res_dir, os.path.split(each_xml_path)[1])
                        shutil.move(each_xml_path, new_xml_path)
                except Exception as e:
                    print(e)
                    print('-' * 50, 'error', '-' * 50)
                    if os.path.exists(each_xml_path):
                        os.remove(each_xml_path)
                self.img_index += 1

            # print process
            if self.print_process:
                use_time = time.time() - self.start_time
                dete_img_num = self.img_index
                dete_speed =  dete_img_num / use_time if dete_img_num > 0 else None
                average_speed = dete_img_num / (time.time() - self.start_time)
                print("* {0} , dete {1} img , speed : {2} pic/second , average speed : {3} pic/second".format(self.img_index, dete_img_num, dete_speed, average_speed))

            # wait
            time.sleep(5)

def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    #
    parser.add_argument('--xml_tmp',dest='xml_tmp',type=str)
    parser.add_argument('--xml_res',dest='xml_res',type=str)
    parser.add_argument('--post_url',dest='post_url',type=str)
    parser.add_argument('--img_count',dest='img_count',type=str)
    parser.add_argument('--mul_progress_num',dest='mul_progress_num',type=int)
    parser.add_argument('--sign_end_txt_dir',dest='sign_end_txt_dir', type=str)
    parser.add_argument('--print_process',dest='print_process', type=str, default='False')
    #
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    ft_server = TZserver(args.xml_tmp, args.xml_res, args.post_url, args.img_count, args.sign_end_txt_dir, args.mul_progress_num)
    ft_server.start_monitor()























