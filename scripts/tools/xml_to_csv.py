# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.utils.FileOperationUtil import FileOperationUtil
from JoTools.utils.CsvUtil import CsvUtil

# todo 查看武汉 csv 是什么样的格式

def xml_to_csv(xml_dir, csv_path):
    csv_list = [['filename', 'code', 'score', 'xmin', 'ymin', 'xmax', 'ymax']]
    for each_xml_path in FileOperationUtil.re_all_file(xml_dir, endswitch=['.xml']):
        each_dete_res = DeteRes(each_xml_path)
        each_img_name = os.path.split(each_dete_res.img_path)[1]
        for dete_obj in each_dete_res:
            csv_list.append([each_img_name, dete_obj.tag, dete_obj.conf, dete_obj.x1, dete_obj.y1, dete_obj.x2, dete_obj.y2])
    CsvUtil.save_list_to_csv(csv_list, csv_path)


if __name__ == "__main__":
    xml_dir = r""
    csv_path = r""
    xml_to_csv(xml_dir, csv_path)















