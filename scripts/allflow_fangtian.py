# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import subprocess
import os
import time
import shutil
from JoTools.utils.FileOperationUtil import FileOperationUtil
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.utils.CsvUtil import CsvUtil

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
mul_process_num = 2
start_time = time.time()
# ----------------------------------------------------------------------------------------------------------------------

class SaveLog():

    def __init__(self, log_path, img_count, csv_path=None):
        self.log_path = log_path
        self.img_count = img_count
        self.img_index = 1
        self.csv_path = csv_path
        self.csv_list = [['filename', 'name', 'score', 'xmin', 'ymin', 'xmax', 'ymax']]
        # empty log
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def add_log(self, img_name):
        self.log = open(self.log_path, 'a')
        self.log.write("process:{0}/{1} {2}\n".format(self.img_index, self.img_count, img_name))
        self.img_index += 1
        self.log.close()

    def add_csv_info(self, dete_res, img_name):
        #
        for dete_obj in dete_res:
            self.csv_list.append([img_name, dete_obj.tag, dete_obj.conf, dete_obj.x1, dete_obj.y1, dete_obj.x2, dete_obj.y2])

    def read_img_list_finshed(self):
        self.log = open(self.log_path, 'a')
        self.log.write("Loading Finished\n")
        self.log.close()

    def close(self):
        self.log = open(self.log_path, 'a')
        self.log.write("---process complete---\n")
        self.log.close()
        # save csv
        CsvUtil.save_list_to_csv(self.csv_list, self.csv_path)

def check_res_file_num(res_txt_dir, txt_num):

    for i in range(1, txt_num+1):
        each_txt_path = os.path.join(res_txt_dir, "{0}.txt".format(i))
        if not os.path.exists(each_txt_path):
            return False
    return True

# --------------------------------------- 001 --------------------------------------------------------------------------

for i in range(1, mul_process_num+1):
    each_cmd_str = r"python3 scripts/all_models_flow.py --scriptIndex {0}-{1} --deteMode {2}".format(mul_process_num, i, dete_mode)
    #each_cmd_str = r"python3 scripts/all_models_flow_mul.py --scriptIndex {0}-{1}".format(mul_process_num, i)
    each_bug_file = open(os.path.join("./logs", "bug{0}_".format(i) + time_str + obj_name + ".txt"), "w+")
    each_std_file = open(os.path.join("./logs", "std1{0}_".format(i) + time_str + obj_name + ".txt"), "w+")
    each_pid = subprocess.Popen(each_cmd_str.split(), stdout=each_std_file, stderr=each_bug_file, shell=False)
    print("pid : {0}".format(each_pid.pid))
    print(each_cmd_str)
    time.sleep(1)
    #cmd = "taskset -a -pc {0} {1}".format('0,1', each_pid.pid)
    #cmd = "taskset -pc {0} {1}".format('0,1', each_pid.pid)
    #os.system(cmd)


# --------------------------------------- 002 --------------------------------------------------------------------------


if os.path.exists(res_txt_dir):
    shutil.rmtree(res_txt_dir)
    os.makedirs(res_txt_dir, exist_ok=True)
else:
    os.makedirs(res_txt_dir, exist_ok=True)

# res_dir
if not os.path.exists(res_dir):
    os.makedirs(res_dir, exist_ok=True)


img_name_dict = {}
img_path_list = list(FileOperationUtil.re_all_file(img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG']))

for each_img_path in img_path_list:
    img_name = FileOperationUtil.bang_path(each_img_path)[1]
    img_name_dict[img_name] = os.path.split(each_img_path)[1]


img_count = len(img_path_list)
save_log = SaveLog(log_path, img_count, csv_path)
save_log.read_img_list_finshed()
#max_use_time = 9.5 * img_count

dete_img_index = 1
while True:

    now_time = time.time()

    # over time
    #if now_time - start_time > max_use_time:
    #    break

    # dete end
    if save_log.img_index > save_log.img_count:
        break
    # all end
    elif check_res_file_num(res_txt_dir, mul_process_num):
        break

    xml_path_list = FileOperationUtil.re_all_file(res_dir, endswitch=['.xml'])

    time.sleep(5)

    for each_xml_path in xml_path_list:
        print("* {0} {1}".format(dete_img_index, each_xml_path))
        img_name = img_name_dict[FileOperationUtil.bang_path(each_xml_path)[1]]
        try:
            # wait for write end
            each_dete_res = DeteRes(each_xml_path)
            img_name = img_name_dict[FileOperationUtil.bang_path(each_xml_path)[1]]
            save_log.add_log(img_name)
            save_log.add_csv_info(each_dete_res, img_name)
            if os.path.exists(each_xml_path):
                os.remove(each_xml_path)
        except Exception as e:
            print(e)
            save_log.add_log(img_name)
            print('-'*50 , 'error' , '-'*50)
            if os.path.exists(each_xml_path):
                os.remove(each_xml_path)
        dete_img_index += 1


save_log.close()

end_time = time.time()
img_count = len(img_path_list)
use_time = end_time - start_time
print("* check img {0} use time {1} {2} s/pic".format(img_count, use_time, use_time/img_count))

#if os.path.exists(res_txt_dir):
#    shutil.rmtree(res_txt_dir)

