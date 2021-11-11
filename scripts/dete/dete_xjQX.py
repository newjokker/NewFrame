# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import cv2
import xml.etree.ElementTree as ET
import configparser
import math
import time

from lib.JoTools.txkjRes.resTools import ResTools
from lib.JoTools.utils.FileOperationUtil import FileOperationUtil
from lib.JoTools.utils.CsvUtil import CsvUtil
from lib.JoTools.txkjRes.deteRes import DeteRes
from lib.JoTools.txkjRes.deteObj import DeteObj
from lib.JoTools.txkjRes.deteAngleObj import DeteAngleObj
from lib.JoTools.utils.JsonUtil import JsonUtil
#
from JoTools.utils.DecoratorUtil import DecoratorUtil
import copy

cwd = os.path.dirname(os.path.abspath(__file__))

@DecoratorUtil.time_this
def dete_xjQX(model_dict, data):
    try:
        model_xjQX_1 = model_dict["model_xjQX_1"]
        # model_xjQX_2 = model_dict["model_xjQX_2"]
        #
        dete_res = model_xjQX_1.detectSOUT(path=data['path'], image=copy.deepcopy(data['im']), image_name=data['name'])
        #
        #dete_res.print_as_fzc_format() 
        dete_res.img_path = data['path']
        dete_res.draw_dete_res(os.path.join('/v0.0.1/tmpfiles', data['name']))

        # todo 线夹个数大于 0 就直接返回 0
        xj_count = 0
        xj_obj = None
        for each_dete_obj in dete_res:
            if 'XJ' in each_dete_obj.tag or 'XianJia' in each_dete_obj.tag:
                xj_count += 1
                xj_obj = each_dete_obj

        if xj_count > 1:
            return DeteRes()
        elif xj_count == 0:
            return DeteRes()

        # todo 将和线夹相交的要素留下，其他的要素全部删除
        dete_res.filter_by_mask(xj_obj.get_points(), cover_index_th=0.0005, need_in=True)
        if len(dete_res) != 2:
            return DeteRes()

        # todo 计算两个要素之间的角度
        obj_1, obj_2 = dete_res[0], dete_res[1]
        angle_1 = obj_1.angle * 180 / math.pi
        angle_2 = obj_2.angle * 180 / math.pi
        xj_res = DeteRes()
        if abs(angle_1 - angle_2) > 100 or abs(angle_1 - angle_2) < 80:
            xj_obj.tag = 'XJfail'
            xj_res.add_obj_2(xj_obj)
        else:
            xj_obj.tag = 'XJnormal'
            xj_res.add_obj_2(xj_obj)

        return xj_res

    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)


