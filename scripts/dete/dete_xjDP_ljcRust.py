# -*- coding: utf-8  -*-
# -*- author: jokker -*-

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
import time
from PIL import Image
import cv2
import numpy as np


@DecoratorUtil.time_this
def dete_xjDP_ljcRust(model_dict, data):
    try:

        # --------------------------------------------------------------------------------------------------------------
        model_xjDP_ljc = model_dict['model_xjDP_ljc']
        model_ljcRust_rust = model_dict['model_ljcRust_rust']
        #
        raw_h, raw_w, _ = data['im'].shape
        detectBoxes = model_xjDP_ljc.detect(data['im'], data['name'])
        #
        dete_res_ljc = DeteRes()
        dete_res_ljc.img_path = data['path']
        dete_res_ljc.file_name = data['name']
        #
        for i, ljj_box in enumerate(detectBoxes):
            points, label = ljj_box[0:4], ljj_box[4]
            resized_name = data['name'] + "_resized_" + label + '_' + str(i)
            xmin, xmax, ymin, ymax = model_xjDP_ljc.minirect2(points, raw_h, raw_w)
            dete_res_ljc.add_obj(int(xmin), int(ymin), int(xmax), int(ymax), str(label), conf=-1, assign_id=i, describe=resized_name)
        #
        dete_res_ljc_big = dete_res_ljc.deep_copy()
        dete_res_ljc_big.filter_by_tags(need_tag=['Sanjiaoban', 'ULuoShuan', 'ZhongChui', 'XuanChuiXianJia', 'DBTZB'])
        dete_res_ljc_big.filter_by_topn(3)
        dete_res_ljc_big.filter_by_area(12000)
        #
        dete_res_ljc_small = dete_res_ljc.deep_copy()
        dete_res_ljc_small.filter_by_tags(need_tag=['PGuaBan', 'ZGuaBan', 'UBGuaBan', 'UGuaHuan', 'ZHGuaHuan', 'ZBDGuaBan', 'WTGB', 'PTTZB','other1'])
        dete_res_ljc_small.filter_by_topn(3)
        dete_res_ljc_small.filter_by_area(6000)
        #
        dete_res_ljc = dete_res_ljc_big + dete_res_ljc_small
        # --------------------------------------------------------------------------------------------------------------
        Area_label = model_ljcRust_rust.get_label()
        #
        for each_dete_obj in dete_res_ljc.alarms:
            each_sub_array = dete_res_ljc.get_sub_img_by_dete_obj(each_dete_obj, RGB=False)
            # OpenCV转换成PIL.Image格式
            img_org = Image.fromarray(cv2.cvtColor(each_sub_array, cv2.COLOR_BGR2RGB))
            # 去除背景
            result = model_ljcRust_rust.detect_image(img_org)                       # 返回的也是Image格式的。所有前景都是1，背景都是0
            img = cv2.cvtColor(np.asarray(img_org), cv2.COLOR_RGB2BGR)
            mask = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
            img_seg = img * mask                                                    # 对应相乘，背景是0 ，所以剩下的就是前景了。
            img_seg = img_seg.astype(np.uint8)
            predict_name = model_ljcRust_rust.cal_rust_matrix(img_seg, Area_label, 0.5)
            each_dete_obj.tag = predict_name
            each_dete_obj.conf = 0.5
        return dete_res_ljc

    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)


