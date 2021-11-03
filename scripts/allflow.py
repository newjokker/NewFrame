# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import subprocess
import os
import time
import shutil
import argparse
from JoTools.utils.FileOperationUtil import FileOperationUtil
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.utils.CsvUtil import CsvUtil


# todo 有一定的概率只启动一个模型，杀掉进程重新启动一下就行，目前不清楚原因


# ----------------------------------------------------------------------------------------------------------------------
img_dir = r"/usr/input_picture"
res_dir = r"/usr/output_dir/save_res"
log_path = r"/usr/output_dir/log"
csv_path = r"/usr/output_dir/result.csv"
res_txt_dir = r"/usr/output_dir/res_txt"
dete_mode = 0
# ----------------------------------------------------------------------------------------------------------------------
obj_name = "_all_flow"
time_str = str(time.time())[:10]
start_time = time.time()
# ----------------------------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--mul_process_num', dest='mul_process_num', type=int, default=2)
    #
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    mul_process_num = args.mul_process_num

    for i in range(1, mul_process_num + 1):
        each_cmd_str = r"python3 scripts/all_models_flow.py --scriptIndex {0}-{1} --deteMode {2}".format(mul_process_num, i, dete_mode)
        each_bug_file = open(os.path.join("./logs", "bug{0}_".format(i) + time_str + obj_name + ".txt"), "w+")
        each_std_file = open(os.path.join("./logs", "std1{0}_".format(i) + time_str + obj_name + ".txt"), "w+")
        each_pid = subprocess.Popen(each_cmd_str.split(), stdout=each_std_file, stderr=each_bug_file, shell=False)
        print("pid : {0}".format(each_pid.pid))
        print(each_cmd_str)
        time.sleep(1)






