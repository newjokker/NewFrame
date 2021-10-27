# -*- coding: utf-8  -*-
# -*- author: jokker -*-

"""
* 定时扫 xml 文件夹，并据此生成 log 和 csv
"""

import os
import time
import shutil
import argparse
from JoTools.utils.FileOperationUtil import FileOperationUtil
from JoTools.txkjRes.deteRes import DeteRes
from .core.save_log import SaveLog


class FTserver(object):

    def __init__(self, img_dir, xml_dir, res_dir, sign_dir, mul_progress_num):
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.res_dir = res_dir
        self.sign_dir = sign_dir
        self.mul_progress_num = mul_progress_num            # 进程个数
        #
        self.start_time = time.time()
        self.stop_time = None
        self.max_dete_time = -1
        self.dete_img_index = 0                             # 已检测的图片的数量
        #
        log_path = os.path.join(res_dir, 'log')
        csv_path = os.path.join(res_dir, 'result.csv')
        img_count = len(list(FileOperationUtil.re_all_file(self.img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG'])))
        self.all_img_count = img_count                                                           # 所有要检测图片的数目
        self.save_log = SaveLog(log_path, img_count, csv_path)
        self.max_dete_time = 9.5 * self.all_img_count

    def if_end(self):
        """根据 sign 文件夹中的信息，判断是否已经结束"""
        for i in range(1, self.mul_progress_num+1):
            each_txt_path = os.path.join(self.sign_dir, "{0}.txt".format(i))
            if not os.path.exists(each_txt_path):
                return False
        return True

    def get_max_dete_time(self, assign_each_dete_use_time=9.5):
        """获取最长的检测时间，超过检测时间自动退出并保存日志和 csv"""
        self.max_dete_time = assign_each_dete_use_time * self.all_img_count

    def empty_history_info(self):
        """清空历史数据"""

        if os.path.exists(self.res_dir):
            shutil.rmtree(self.res_dir)

        if os.path.exists(self.sign_dir):
            shutil.rmtree(self.sign_dir)

        if os.path.exists(self.xml_dir):
            shutil.rmtree(self.xml_dir)

        os.makedirs(self.res_dir, exist_ok=True)
        os.makedirs(sign_dir, exist_ok=True)
        os.makedirs(xml_dir, exist_ok=True)

    def start_monitor(self):
        """开始监听"""
        img_name_dict = {}
        while True:
            # dete end
            if self.save_log.img_index > self.save_log.img_count:
                self.stop_time = time.time()
                return

            # all end
            elif self.if_end():
                self.stop_time = time.time()
                return

            xml_path_list = FileOperationUtil.re_all_file(res_dir, endswitch=['.xml'])

            time.sleep(3)

            for each_xml_path in xml_path_list:
                print("* {0} {1}".format(self.dete_img_index + 1, each_xml_path))
                img_name = img_name_dict[FileOperationUtil.bang_path(each_xml_path)[1]]
                try:
                    # wait for write end
                    each_dete_res = DeteRes(each_xml_path)
                    img_name = img_name_dict[FileOperationUtil.bang_path(each_xml_path)[1]]
                    self.save_log.add_log(img_name)
                    self.save_log.add_csv_info(each_dete_res, img_name)
                    if os.path.exists(each_xml_path):
                        os.remove(each_xml_path)
                except Exception as e:
                    print(e)
                    self.save_log.add_log(img_name)
                    print('-' * 50, 'error', '-' * 50)
                    if os.path.exists(each_xml_path):
                        os.remove(each_xml_path)
                self.dete_img_index += 1

    def get_csv(self):
        self.save_log.close()
        use_time = self.stop_time - self.start_time
        print("* check img {0} use time {1} {2} s/pic".format(self.all_img_count, use_time, use_time / self.all_img_count))

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
    parser.add_argument('--sign_dir',dest='rsign_dir',type=str, default=r"/usr/input_picture")
    parser.add_argument('--mul_progress_num',dest='mul_progress_num',type=str, default=1)
    #
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    ft_server = FTserver(args.img_dir, args.xml_dir, args.res_dir, args.sign_dir, args.mul_progress_num)
    ft_server.main()























