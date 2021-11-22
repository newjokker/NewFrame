# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import time
import argparse
import requests
from multiprocessing import Pool
from JoTools.utils.FileOperationUtil import FileOperationUtil
from JoTools.txkjRes.deteRes import DeteRes


def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--port', dest='port', type=int, default=3232)
    parser.add_argument('--host', dest='host', type=str, default='192.168.3.74')  # 这边要是写 127 的话只能在本服务器访问了，要改为本机的地址
    #
    parser.add_argument('--xml_dir', dest='xml_dir', type=str, default='/')
    parser.add_argument('--post_mode', dest='post_mode', type=str, default=0)
    #
    args = parser.parse_args()
    return args


def post_dete_res(assign_dete_res):
    """推送结果"""
    url = r"http://{0}:{1}/sendResult".format(host, port)
    data = []
    for each_dete_res in assign_dete_res:
        data.append({
            "file_name": each_dete_res.file_name,
            # "result": each_dete_res.get_fzc_format(),
            "result": each_dete_res.get_result_construction(),
            "p_id": each_dete_res.p_id
        })
    res = requests.post(url=url, data=data)
    print(res)


if __name__ == "__main__":

    args = parse_args()
    host = args.host
    port = args.port
    xml_dir = args.xml_dir
    post_mode = args.post_mode

    while True:
        dete_res_all = []
        for each_xml_path in FileOperationUtil.re_all_file(xml_dir, endswitch=['.xml']):
            each_dete_res = DeteRes(xml_path=each_xml_path)
            each_dete_res.file_name = FileOperationUtil.bang_path(each_xml_path)[1]
            each_dete_res.p_id = -1
            dete_res_all.append(each_dete_res)
            # remove
            os.remove(each_xml_path)
        # post
        post_dete_res(dete_res_all)
        time.sleep(20)


