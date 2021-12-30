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
res_dir = r"/usr/output_dir"
res_xml_tmp = r"/usr/output_dir/xml_tmp"
res_xml_res = r"/usr/output_dir/xml_res"
log_path = r"/usr/output_dir/log"
log_dir = r"/usr/output_dir/logs"
os.makedirs(log_dir, exist_ok=True)
csv_path = r"/usr/output_dir/result.csv"
sign_dir = r"/v0.0.1/sign"
res_txt_dir = r"/usr/output_dir/res_txt"
receive_post_config_path = r"/v0.0.1/sign/receive_post_config.ini"
picture_name_json_path = r"/usr/input_picture_attach/pictureName.json"
# ----------------------------------------------------------------------------------------------------------------------
obj_name = "_all_flow"
time_str = str(time.time())[:10]
start_time = time.time()
# ----------------------------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--mul_process_num', dest='mul_process_num', type=int, default=2)
    parser.add_argument('--gpu_id_list', dest='gpu_id_list', type=str, default='0')
    parser.add_argument('--model_list', dest='model_list', type=str, default='')
    parser.add_argument('--model_types', dest='model_types', type=str, default='01,02,03')
    #
    args = parser.parse_args()
    return args

def start_ft_server(xml_dir, img_dir, res_dir, sign_dir, mul_process_num=2):
    ft_server_cmd_str = r"python3 /v0.0.1/scripts/server/fangtian/fangtian_server.py --img_dir {0} --xml_dir {1} --res_dir {2} --sign_dir {3} --mul_progress_num {4} --picture_name_json_path {5}".format(
        img_dir, xml_dir, res_dir, sign_dir, mul_process_num, picture_name_json_path)
    print(ft_server_cmd_str)
    receive_pid = subprocess.Popen(ft_server_cmd_str.split(), shell=False)
    return receive_pid



if __name__ == "__main__":

    # fixme 对比的是 经过 ft_server 处理过，的那些 xml ，


    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()
    mul_process_num = args.mul_process_num
    model_list_str = args.model_list
    model_list = model_list_str.strip().split(',')
    gpu_id_list_str = args.gpu_id_list
    gpu_id_list = gpu_id_list_str.strip().split(',')
    gpu_num = len(gpu_id_list)
    model_types = args.model_types.strip().split(',')
    # ------------------------------------------------------------------------------------------------------------------
    # filter model_list
    # del_model_list = []
    # if '01' not in model_types:
    #     del_model_list.append('nc')
    # if '03' not in model_types:
    #     del_model_list.append('jyzZB')
    # if '04' not in model_types:
    #     del_model_list.append("kkxTC")
    #     del_model_list.append("kkxQuiting")
    #     del_model_list.append("kkxClearance")
    #     del_model_list.append("fzc")
    #     del_model_list.append("fzcRust")
    #     #
    #     del_model_list.append("xjDP")
    #     del_model_list.append("ljcRust")
    # #
    # for each_del_model in del_model_list:
    #     if each_del_model in model_list:
    #         model_list.remove(each_del_model)
    # ------------------------------------------------------------------------------------------------------------------

    # if no model to dete : exit
    if len(model_list) == 0:
        print("* no model to dete")
        exit()
    else:
        print("* dete model include : {0}".format(', '.join(model_list)))

    # empty history
    if os.path.exists(res_xml_tmp):
        for each_file_path in FileOperationUtil.re_all_file(res_xml_tmp):
            os.remove(each_file_path)

    # start dete servre
    for i in range(1, mul_process_num + 1):
        each_cmd_str = r"python3 scripts/all_models_flow.py --scriptIndex {0}-{1} --gpuID {2} --model_list {3} --assign_img_dir {4} --ignore_history {5} --del_dete_img {6}".format(
            mul_process_num, i, gpu_id_list[(i-1)%gpu_num], ','.join(model_list), r'/usr/input_picture', 'True', 'False')

        each_bug_file = open(os.path.join(log_dir, "bug{0}_".format(i) + time_str + obj_name + ".txt"), "w+")
        each_std_file = open(os.path.join(log_dir, "std1{0}_".format(i) + time_str + obj_name + ".txt"), "w+")

        each_pid = subprocess.Popen(each_cmd_str.split(), stdout=each_std_file, stderr=each_bug_file, shell=False)
        print("pid : {0}".format(each_pid.pid))
        print("* {0}".format(each_cmd_str))
        time.sleep(1)

    # start fangtian server
    start_ft_server(res_xml_tmp, img_dir, res_dir, sign_dir, mul_process_num)

    # 等待
    while True:
        time.sleep(60)

    # ------------------------------------------------------------------------------------------------------------------

    # # todo check and print dete res
    # os.makedirs(res_xml_res, exist_ok=True)
    # start_time = time.time()
    # img_path_list = list(FileOperationUtil.re_all_file(img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG']))
    # img_count = len(img_path_list)
    # # 用于在断点续传的时候准确计算检测速度
    # xml_count_region = len(list(FileOperationUtil.re_all_file(res_xml_res, endswitch=['.xml'])))
    #
    # while True:
    #     res_xml_list = list(FileOperationUtil.re_all_file(res_xml_res, endswitch=['.xml']))
    #     xml_count = len(res_xml_list)
    #     if xml_count >= img_count:
    #         print("* detection finished")
    #         break
    #     else:
    #         use_time = time.time()-start_time
    #         print("* detection : {0} | {1} | {2} | {3}s/pic".format(xml_count, img_count-xml_count, use_time, use_time / (max(xml_count - xml_count_region, 1))))
    #         time.sleep(30)

    # ------------------------------------------------------------------------------------------------------------------
    # print("* xml to csv")
    # xml_to_csv(res_xml_res, csv_path)
    # print("* xml to csv success ")

    # use_time_all = time.time() - start_time
    # print("* dete use time : {0} ,  {1}s/pic".format(use_time_all, use_time_all/img_count))


