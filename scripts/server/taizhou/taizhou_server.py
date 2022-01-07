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


# todo 图像先进行一次解析，不能出现一样的文件名

# test taizhou_server.py
# python3 scripts/server/taizhou/taizhou_server.py --xml_tmp /home/ldq/NewFrame_TaiZhou/tmpfiles/xml_tmp --xml_res /home/ldq/NewFrame_TaiZhou/tmpfiles/xml_res --post_url 192.168.3.000 --img_count 10 --mul_progress_num 1 --sign_end_txt_dir /home/ldq/NewFrame_TaiZhou/tmpfiles/all_models_flow --print_process True

# test all_model_flow.py
# python3 scripts/all_models_flow.py --scriptIndex 1-1 --gpuID 0 --model_list nc --assign_img_dir /home/ldq/NewFrame_TaiZhou/test_img_path.txt --ignore_history True --del_dete_img False --signDir /home/ldq/NewFrame_TaiZhou/tmpfiles --outputDir /home/ldq/NewFrame_TaiZhou/tmpfiles


class TZserver(object):

    def __init__(self, xml_tmp, xml_res, post_url, img_count, sign_dir, mul_progress_num, batch_id, print_proecss="True"):
        self.xml_tmp = xml_tmp
        self.xml_res = xml_res
        self.post_url = post_url
        self.img_count = int(img_count)
        self.img_index = 0
        #
        self.sign_dir = sign_dir                # 用于监测模型是不是已经运行成功了，状态文件夹
        #
        self.print_process = eval(print_proecss)
        #
        self.mul_progress_num = mul_progress_num
        #
        self.start_time = time.time()
        #
        self.batch_id = batch_id
        #
        self.model_end = False

    def if_end(self):
        """根据 sign 文件夹中的信息，判断是否已经结束 | 根据已检测的图片和图片的总数是否相等"""

        if self.img_index >= self.img_count:
            return True

        for i in range(1, self.mul_progress_num+1):
            each_txt_path = os.path.join(self.sign_dir, 'res_txt', "{0}.txt".format(i))
            if not os.path.exists(each_txt_path):
                # 模型结束之后只让处理最后一次的 xml
                if self.model_end:
                    return False
                else:
                    self.model_end = True
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
            "alarms": [],
            "batch_id":self.batch_id
        }
        # alarms
        alarms, id = [], 0
        for each_dete_obj in dete_res:
            id += 1
            alarms.append([id, each_dete_obj.x1, each_dete_obj.y1, each_dete_obj.x2, each_dete_obj.y2, each_dete_obj.tag, each_dete_obj.conf])
        # time
        if dete_res.des in ["", None]:
            start_time, end_time = -1, -1
        else:
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
            # response_data = requests.post(url=self.post_url,  data=json.dumps(data), headers=headers)
            print('-'*50)
            print("* img index {0} post : {1} data : {2}".format(self.img_index, self.post_url, data))
            # return response_data
        except Exception as e:
            print("----error----")
            print(e)

    def start_monitor(self):
        """开始监听"""
        #
        while True:
            if self.if_end():
                print("----- server finished ------")
                return

            # ----------------------------------------------------------------------------------------------------------
            xml_path_list = list(FileOperationUtil.re_all_file(self.xml_tmp, endswitch=['.xml']))
            #
            for each_xml_path in xml_path_list:
                # fixme post error, wait for next post | or just skip
                try:
                    self.post_res(each_xml_path)
                except Exception as e:
                    print(e)
                    print(e.__traceback__.tb_frame.f_globals["__file__"])
                    print(e.__traceback__.tb_lineno)
                    continue
                #
                if os.path.exists(each_xml_path):
                    new_xml_path = os.path.join(self.xml_res, os.path.split(each_xml_path)[1])
                    shutil.move(each_xml_path, new_xml_path)
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
            time.sleep(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    #
    parser.add_argument('--xml_tmp',dest='xml_tmp',type=str)
    parser.add_argument('--xml_res',dest='xml_res',type=str)
    parser.add_argument('--post_url',dest='post_url',type=str)
    parser.add_argument('--img_count',dest='img_count',type=str)
    parser.add_argument('--mul_progress_num',dest='mul_progress_num', type=int)
    parser.add_argument('--sign_end_txt_dir',dest='sign_end_txt_dir', type=str)
    parser.add_argument('--print_process',dest='print_process', type=str, default='False')
    parser.add_argument('--batch_id',dest='batch_id', type=str, default='--no_batch_id--')
    #
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    ft_server = TZserver(args.xml_tmp, args.xml_res, args.post_url, args.img_count, args.sign_end_txt_dir, args.mul_progress_num, args.batch_id, "True")
    ft_server.start_monitor()























