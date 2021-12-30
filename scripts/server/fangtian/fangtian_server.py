# -*- coding: utf-8  -*-
# -*- author: jokker -*-

"""
* 定时扫 xml 文件夹，并据此生成 log 和 csv
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

    def init_log(self):
        log = open(self.log_path, 'a')
        log.write("Loading Finished\n")
        log.close()

    def add_log(self, img_name):
        log = open(self.log_path, 'a')
        log.write("process:{0}/{1} {2}\n".format(self.img_index, self.img_count, img_name))
        self.img_index += 1
        log.close()

    def add_csv_info(self, dete_res, img_name):
        """add one csv info"""
        csv_list= []
        for dete_obj in dete_res:
            csv_list.append([img_name, dete_obj.tag, dete_obj.conf, dete_obj.x1, dete_obj.y1, dete_obj.x2, dete_obj.y2])

        with open(self.csv_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_list)

    def close(self):
        log = open(self.log_path, 'a')
        log.write("---process complete---\n")
        log.close()
        # save csv
        # CsvUtil.save_list_to_csv(self.csv_list, self.csv_path)


class FTserver(object):

    def __init__(self, img_dir, xml_dir, res_dir, sign_dir, mul_progress_num, picture_name_json_path, pid_list=None, print_proecss="True"):
        # fixme 用 fangtian_server 管理跑的 pid ，跑完了直接关掉对应的 pid
        self.img_dir = img_dir
        self.xml_tmp_dir = xml_dir
        self.res_dir = res_dir
        self.xml_res_dir = os.path.join(res_dir, "xml_res")
        self.sign_dir = sign_dir
        self.sign_end_txt_dir = os.path.join(self.sign_dir, 'res_txt')
        self.mul_progress_num = mul_progress_num            # 进程个数
        #
        self.start_time = time.time()
        self.last_flash_time = self.start_time
        self.stop_time = None
        self.max_dete_time = -1
        self.dete_img_index = 0                             # 已检测的图片的数量
        #
        log_path = os.path.join(res_dir, 'log')
        csv_path = os.path.join(res_dir, 'result.csv')
        img_count = len(list(FileOperationUtil.re_all_file(self.img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG'])))
        self.all_img_count = img_count                                                           # 所有要检测图片的数目
        self.save_log = SaveLog(log_path, img_count, csv_path)
        self.save_log.init_log()
        self.max_dete_time = 9.5 * self.all_img_count
        #
        self.picture_name_json_path = picture_name_json_path
        self.img_name_json_dict = {}
        #
        self.print_process = eval(print_proecss)
        # 解析 json 文件
        #self.parse_json_dict()

    def if_end(self):
        """根据 sign 文件夹中的信息，判断是否已经结束"""
        for i in range(1, self.mul_progress_num+1):
            each_txt_path = os.path.join(self.sign_end_txt_dir, "{0}.txt".format(i))
            if not os.path.exists(each_txt_path):
                return False
        return True

    def parse_json_dict(self):
        name_info = JsonUtil.load_data_from_json_file(self.picture_name_json_path)
        for each in name_info:
            self.img_name_json_dict[each["fileName"]] = each["originFileName"]

    def get_max_dete_time(self, assign_each_dete_use_time=9.5):
        """获取最长的检测时间，超过检测时间自动退出并保存日志和 csv"""
        self.max_dete_time = assign_each_dete_use_time * self.all_img_count

    def empty_dir(self, assign_dir):
        """清空文件夹中的文件"""
        for each_file_path in FileOperationUtil.re_all_file(assign_dir):
            os.remove(each_file_path)

    def empty_history_info(self):
        """清空历史数据"""

        if os.path.exists(self.sign_end_txt_dir):
            self.empty_dir(self.sign_end_txt_dir)

        os.makedirs(self.res_dir, exist_ok=True)
        os.makedirs(self.xml_res_dir, exist_ok=True)
        os.makedirs(self.sign_dir, exist_ok=True)
        os.makedirs(self.xml_tmp_dir, exist_ok=True)

    def start_monitor(self):
        """开始监听"""
        # store img name and suffix
        img_name_dict = {}
        img_path_list = list(FileOperationUtil.re_all_file(self.img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG']))
        for each_img_path in img_path_list:
            img_name = FileOperationUtil.bang_path(each_img_path)[1]
            img_name_dict[img_name] = os.path.split(each_img_path)[1]

        # --------------------------------------------------------------------------------------------------------------
        # 先将已经完成的结果数据放到 log 中去，用于断点检测
        xml_list_new = FileOperationUtil.re_all_file(self.xml_res_dir, endswitch=['.xml'])
        for each_xml_path in xml_list_new:
            img_name = img_name_dict[FileOperationUtil.bang_path(each_xml_path)[1]]
            try:
                # wait for write end
                each_dete_res = DeteRes(each_xml_path)
                img_name = img_name_dict[FileOperationUtil.bang_path(each_xml_path)[1]]
                self.save_log.add_log(img_name)
                self.save_log.add_csv_info(each_dete_res, img_name)
            except Exception as e:
                print(e)
                self.save_log.add_log(img_name)
                print('-' * 50, 'error', '-' * 50)
                if os.path.exists(each_xml_path):
                    os.remove(each_xml_path)
            self.dete_img_index += 1

        # update start time
        self.start_time = time.time()

        # --------------------------------------------------------------------------------------------------------------

        #
        while True:

            # todo 设置多种退出机制（1）超时（2）根据 end_txt_dir 中的 txt 证明结束 （3）结果 xml 数目大于等于图片数目
            # dete end
            if self.save_log.img_index > self.save_log.img_count:
                self.stop_time = time.time()
                return

            # all end
            elif self.if_end():
                self.stop_time = time.time()
                return

            xml_path_list = list(FileOperationUtil.re_all_file(self.xml_tmp_dir, endswitch=['.xml']))
            #
            for each_xml_path in xml_path_list:
                img_name = img_name_dict[FileOperationUtil.bang_path(each_xml_path)[1]]
                try:
                    each_dete_res = DeteRes(each_xml_path)
                    self.save_log.add_log(img_name)
                    self.save_log.add_csv_info(each_dete_res, img_name)
                    if os.path.exists(each_xml_path):
                        new_xml_path = os.path.join(self.xml_res_dir, os.path.split(each_xml_path)[1])
                        shutil.move(each_xml_path, new_xml_path)
                except Exception as e:
                    print(e)
                    self.save_log.add_log(img_name)
                    print('-' * 50, 'error', '-' * 50)
                    if os.path.exists(each_xml_path):
                        os.remove(each_xml_path)
                self.dete_img_index += 1

            # print process
            if self.print_process:
                use_time = time.time() - self.last_flash_time
                self.last_flash_time = time.time()
                dete_img_num = len(xml_path_list)
                dete_speed =  dete_img_num / use_time if dete_img_num > 0 else None
                average_speed = self.dete_img_index / (time.time()-self.start_time)
                print("* {0} , dete {1} img , speed : {2} pic/second , average speed : {3} pic/second".format(self.dete_img_index, dete_img_num, dete_speed, average_speed))

            # wait
            time.sleep(5)


    def get_csv(self):
        use_time = self.stop_time - self.start_time
        print("* check img {0} use time {1} {2} s/pic".format(self.all_img_count, use_time, use_time / self.all_img_count))
        print("* xml to csv")
        self.save_log.close()
        print("* xml to csv success ")

    def main(self):
        """主流程"""
        self.empty_history_info()
        self.get_max_dete_time(9.5)
        self.start_monitor()
        self.get_csv()


def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    #
    parser.add_argument('--img_dir',dest='img_dir',type=str, default=r"/usr/input_picture")
    parser.add_argument('--xml_dir',dest='xml_dir',type=str, default=r"/usr/input_picture")
    parser.add_argument('--res_dir',dest='res_dir',type=str, default=r"/usr/input_picture")
    parser.add_argument('--sign_dir',dest='sign_dir',type=str, default=r"/usr/input_picture")
    parser.add_argument('--mul_progress_num',dest='mul_progress_num',type=int, default=1)
    parser.add_argument('--picture_name_json_path',dest='picture_name_json_path', type=str, default=r'/usr/input_picture_attach/pictureName.json')
    parser.add_argument('--print_process',dest='print_process', type=str, default='False')
    #
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # fixme 删除历史文件
    #

    args = parse_args()
    ft_server = FTserver(args.img_dir, args.xml_dir, args.res_dir, args.sign_dir, args.mul_progress_num, args.picture_name_json_path)
    ft_server.main()























