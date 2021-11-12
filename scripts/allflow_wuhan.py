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
#
from tools.xml_to_csv import xml_to_csv

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


    # todo 不要删除结果，只是将结果 xml 转移到另外一个地方而已 [√]

    # todo 只是实时进行统计有多少 xml，不用一直进行删除等操作 [√]

    # todo 要有办法能实时显示已经跑完了多少数据，这些数据中一共有多少的目标，有多少有目标的图片

    # todo 最后生成的 excel 样式和方天的不一样，一定要重新修改一下 [√]

    # todo 断点继续检测 [√]

    # ------------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------------

    # todo check and print dete res
    start_time = time.time()
    img_path_list = list(FileOperationUtil.re_all_file(img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG']))
    img_count = len(img_path_list)

    while True:
        res_xml_list = list(FileOperationUtil.re_all_file(res_dir, endswitch=['.xml']))
        xml_count = len(res_xml_list)
        if xml_count >= img_count:
            print("* detection finished")
            break
        else:
            use_time = time.time()-start_time
            print("* detection : {0} | {1} | {2} | {3}s/pic".format(xml_count, img_count-xml_count, use_time, use_time / max(xml_count, 1)))
            time.sleep(30)

    # ------------------------------------------------------------------------------------------------------------------
    print("* xml to csv")
    xml_to_csv(res_dir, csv_path)
    print("* xml to csv success ")

    use_time_all = time.time() - start_time
    print("* dete use time : {0} ,  {1}s/pic".format(use_time_all, use_time_all/img_count))






