# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
from JoTools.utils.CsvUtil import CsvUtil


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
        log = open(self.log_path, 'a')
        log.write("process:{0}/{1} {2}\n".format(self.img_index, self.img_count, img_name))
        self.img_index += 1
        log.close()

    def add_csv_info(self, dete_res, img_name):
        """add one csv info"""
        for dete_obj in dete_res:
            self.csv_list.append([img_name, dete_obj.tag, dete_obj.conf, dete_obj.x1, dete_obj.y1, dete_obj.x2, dete_obj.y2])

    def read_img_list_finshed(self):
        log = open(self.log_path, 'a')
        log.write("Loading Finished\n")
        log.close()

    def close(self):
        log = open(self.log_path, 'a')
        log.write("---process complete---\n")
        log.close()
        # save csv
        CsvUtil.save_list_to_csv(self.csv_list, self.csv_path)


if __name__ == "__main__":

    pass









