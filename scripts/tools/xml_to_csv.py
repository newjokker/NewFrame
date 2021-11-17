# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import sys
import argparse
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.utils.FileOperationUtil import FileOperationUtil
from JoTools.utils.CsvUtil import CsvUtil
from JoTools.utils.PrintUtil import PrintUtil


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='run model')
    parser.add_argument('--xml_dir', dest='xml_dir',type=str)
    parser.add_argument('--csv_path', dest='csv_path',type=str)
    assign_args = parser.parse_args()
    return assign_args


def xml_to_csv(xml_dir, csv_path):
    csv_list = [['filename', 'code', 'score', 'xmin', 'ymin', 'xmax', 'ymax']]
    for each_xml_path in FileOperationUtil.re_all_file(xml_dir, endswitch=['.xml']):
        try:
            each_dete_res = DeteRes(each_xml_path)
            each_img_name = os.path.split(each_dete_res.img_path)[1]
            for dete_obj in each_dete_res:
                csv_list.append([each_img_name, dete_obj.tag, dete_obj.conf, dete_obj.x1, dete_obj.y1, dete_obj.x2, dete_obj.y2])
        except Exception as e:
            print(e)
    CsvUtil.save_list_to_csv(csv_list, csv_path)


if __name__ == "__main__":


    if len(sys.argv) > 1:
        args = parse_args()
        PrintUtil.print(args)
        xml_to_csv(args.xml_dir, args.csv_path)
    else:
        xml_dir = r""
        csv_path = r""
        xml_to_csv(xml_dir, csv_path)















