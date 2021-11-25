# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import subprocess
import os
import time
import shutil
import argparse
import configparser
from JoTools.utils.FileOperationUtil import FileOperationUtil
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.utils.CsvUtil import CsvUtil
#
from tools.xml_to_csv import xml_to_csv

# todo 有一定的概率只启动一个模型，杀掉进程重新启动一下就行，目前不清楚原因

# ----------------------------------------------------------------------------------------------------------------------
img_dir = r"/usr/input_picture"
res_dir = r"/usr/output_dir/save_res"
log_path = r"/usr/output_dir/log"
csv_path = r"/usr/output_dir/result.csv"
res_txt_dir = r"/usr/output_dir/res_txt"
receive_post_config_path = r"/v0.0.1/sign/receive_post_config.ini"
# ----------------------------------------------------------------------------------------------------------------------
obj_name = "_all_flow"
time_str = str(time.time())[:10]
start_time = time.time()
# ----------------------------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--mul_process_num', dest='mul_process_num', type=int, default=2)
    parser.add_argument('--gpu_id_list', dest='gpu_id_list', type=str, default='0')
    parser.add_argument('--dete_mode', dest='dete_mode', type=int, default=1)
    parser.add_argument('--model_list', dest='model_list', type=str, default='')
    #
    args = parser.parse_args()
    return args

def start_receive_server(config_path):
    cf = configparser.ConfigParser()
    cf.read(config_path)
    #
    if cf.has_section('receive'):
        receive_host = cf.get('receive', 'host')
        receive_port = cf.get('receive', 'port')
        receive_img_dir = cf.get('receive', 'img_dir')
        receive_batch_size = cf.get('receive', 'batch_size')
        #
        receive_cmd_str = r"python3 scripts/server/hot_wheel/hot_wheel_receive_server.py --host {0} --port {1} --img_dir {2} --batch_size {3}".format(
            receive_host, receive_port, receive_img_dir, receive_batch_size)
        print(receive_cmd_str)
        each_bug_file = open(os.path.join("./logs", "bug_receive_" + time_str + obj_name + ".txt"), "w+")
        each_std_file = open(os.path.join("./logs", "std_receive_" + time_str + obj_name + ".txt"), "w+")
        receive_pid = subprocess.Popen(receive_cmd_str.split(), stdout=each_std_file, stderr=each_bug_file, shell=False)
        return receive_pid
    else:
        return None

def start_post_server(config_path):
    cf = configparser.ConfigParser()
    cf.read(config_path)
    #
    if cf.has_section('post'):
        post_host = cf.get('post', 'host')
        post_port = cf.get('post', 'port')
        post_xml_dir = cf.get('post', 'xml_dir')
        post_mode = cf.get('post', 'post_mode')
        #
        post_cmd_str = r"python3 scripts/server/hot_wheel/hot_wheel_post_server.py --host {0} --port {1} --xml_dir {2} --post_mode {3}".format(
            post_host, post_port, post_xml_dir, post_mode)
        print(post_cmd_str)
        each_bug_file = open(os.path.join("./logs", "bug_post_" + time_str + obj_name + ".txt"), "w+")
        each_std_file = open(os.path.join("./logs", "std_post_" + time_str + obj_name + ".txt"), "w+")
        post_pid = subprocess.Popen(post_cmd_str.split(), stdout=each_std_file, stderr=each_bug_file, shell=False)
        return post_pid
    else:
        return None


if __name__ == "__main__":

    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()
    mul_process_num = args.mul_process_num
    dete_mode = args.dete_mode
    model_list_str = args.model_list
    model_list = model_list_str.strip().split(',')
    gpu_id_list_str = args.gpu_id_list
    gpu_id_list = gpu_id_list_str.strip().split(',')
    gpu_num = len(gpu_id_list)

    # if no model to dete : exit
    if len(model_list) == 0:
        print("* no model to dete")
        exit()
    else:
        print("* dete model include : {0}".format(', '.join(model_list)))

    # start receive server and post server
    if os.path.exists(receive_post_config_path):
        start_receive_server(receive_post_config_path)
        start_post_server(receive_post_config_path)
    else:
        print("* no receive_post_config.ini just run core dete")

    # start dete servre
    for i in range(1, mul_process_num + 1):
        each_cmd_str = r"python3 scripts/all_models_flow.py --scriptIndex {0}-{1} --deteMode {2} --gpuID {3} --model_list {4} --keep_alive {5} --ignore_history {6}".format(
            mul_process_num, i, dete_mode, gpu_id_list[(i-1)%gpu_num], ','.join(model_list), 'True', 'False')

        each_bug_file = open(os.path.join("./logs", "bug{0}_".format(i) + time_str + obj_name + ".txt"), "w+")
        each_std_file = open(os.path.join("./logs", "std1{0}_".format(i) + time_str + obj_name + ".txt"), "w+")

        each_pid = subprocess.Popen(each_cmd_str.split(), stdout=each_std_file, stderr=each_bug_file, shell=False)
        print("pid : {0}".format(each_pid.pid))
        print("* {0}".format(each_cmd_str))
        time.sleep(1)

    while True:
        time.sleep(1)






