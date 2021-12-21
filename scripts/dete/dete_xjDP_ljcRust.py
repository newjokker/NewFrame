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
        model_xjDP_kkx = model_dict['model_xjDP_kkx']
        model_xjDP_cls = model_dict['model_xjDP_cls']
        #
        # --------------------------------------------------------------------------------------------------------------

        dete_res_ljc = model_xjDP_ljc.detectSOUT(path=data['path'], image=copy.deepcopy(data['im']), image_name=data['name'])
        dete_res_ljc.do_nms(threshold=0.5)
        dete_res_ljc.img_path = data['path']
        dete_res_ljc.file_name = data['name']
        dete_res_ljc.img_ndarry = data['im']

        # --------------------------------------------------------------------------------------------------------------
        # ljcRust
        dete_res_ljc_big = dete_res_ljc.deep_copy(copy_img=False)
        dete_res_ljc_big.filter_by_tags(need_tag=['Sanjiaoban', 'ULuoShuan', 'ZhongChui', 'XuanChuiXianJia', 'DBTZB'])
        dete_res_ljc_big.filter_by_topn(3)
        dete_res_ljc_big.filter_by_area(12000)
        #
        dete_res_ljc_small = dete_res_ljc.deep_copy(copy_img=False)
        dete_res_ljc_small.filter_by_tags(need_tag=['PGuaBan', 'ZGuaBan', 'UBGuaBan', 'UGuaHuan', 'ZHGuaHuan', 'ZBDGuaBan', 'WTGB', 'PTTZB','other1'])
        dete_res_ljc_small.filter_by_topn(3)
        dete_res_ljc_small.filter_by_area(6000)
        #
        dete_res_ljc = dete_res_ljc_big + dete_res_ljc_small
        # --------------------------------------------------------------------------------------------------------------
        Area_label = model_ljcRust_rust.get_label()
        #
        for each_dete_obj in dete_res_ljc.alarms:
            each_sub_array = dete_res_ljc.get_sub_img_by_dete_obj_new(each_dete_obj, RGB=False)
            # OpenCV转换成PIL.Image格式
            img_org = Image.fromarray(cv2.cvtColor(each_sub_array, cv2.COLOR_BGR2RGB))
            # 去除背景
            result = model_ljcRust_rust.detect_image(img_org)                       # 返回的也是Image格式的。所有前景都是1，背景都是0
            img = cv2.cvtColor(np.asarray(img_org), cv2.COLOR_RGB2BGR)
            mask = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
            img_seg = img * mask                                                    # 对应相乘，背景是0 ，所以剩下的就是前景了。
            img_seg = img_seg.astype(np.uint8)
            predict_name = model_ljcRust_rust.cal_rust_matrix(img_seg, Area_label, 0.5)
            each_dete_obj.tag = each_dete_obj.tag + '_' + predict_name
            each_dete_obj.conf = 0.5
        # --------------------------------------------------------------------------------------------------------------
        # xjDP_kkx
        dete_kg_lm = dete_res_ljc.deep_copy(copy_img=False)
        dete_kg_lm.reset_alarms([])
        # 遍历每一个连接件正框
        for each_dete_obj in dete_res_ljc:
            each_dete_kg_lm = dete_res_ljc.deep_copy(copy_img=False)
            ## kkg+others on a ljc cap ##
            each_dete_kg_lm.reset_alarms([])
            # get array 连接件正框图片矩阵 np.array
            each_sub_array = dete_res_ljc.get_sub_img_by_dete_obj_new(each_dete_obj, RGB=True, augment_parameter=[0.2, 0.2, 1.2, 0.2])
            ljc_yCenter = int((each_dete_obj.y1 + each_dete_obj.y2)*0.5)
            # 小金具定位检测结果集合 on a ljc martrix-cap
            out = model_xjDP_kkx.detect(each_sub_array, data['name'])
            if len(out[0]) > 0:
                voc_labels = model_xjDP_kkx.post_process(*out)
                ## 过滤最小尺寸 ##
                voc_labels = model_xjDP_kkx.checkDetectBoxAreas(voc_labels)

                for each_obj in voc_labels:
                    ## label, i, xmin, ymin, xmax, ymax,p
                    new_dete_obj = DeteObj(each_obj[2], each_obj[3], each_obj[4], each_obj[5], tag=each_obj[0], conf=float(each_obj[6]), assign_id=each_dete_obj.id)
                    each_dete_kg_lm.add_obj_2(new_dete_obj)

                ## +xmap +ymap 坐标还原至原图
                each_dete_kg_lm.offset(each_dete_obj.x1, each_dete_obj.y1)
                each_dete_dp = each_dete_kg_lm.deep_copy(copy_img=False)

                each_dete_dp.reset_alarms([])
                y_list = [each_dete_obj.y1  for each_dete_obj in each_dete_kg_lm if each_dete_obj.tag in ['K', 'KG','K2', 'KG2']]

                if len(y_list) > 0:
                    ymax = max(y_list)
                    for each_dete_obj in each_dete_kg_lm:
                        if ymax == each_dete_obj.y1 and each_dete_obj.y1 > ljc_yCenter:
                            each_dete_dp.add_obj_2(each_dete_obj)
                            break
                    # merge
                    dete_kg_lm += each_dete_dp

        # gloabal nms
        dete_kg_lm.do_nms_in_assign_tags(tag_list=['K', 'KG', 'Lm', 'K2', 'KG2'],threshold=0.3)

        '''业务逻辑：other* 和 dense内的K过滤'''
        dete_res_kkx = dete_kg_lm.deep_copy(copy_img=False)
        dete_res_kkx.img_ndarry = data['im']
        #
        only_other_3 = dete_kg_lm.deep_copy(copy_img=False)
        only_k = dete_kg_lm.deep_copy(copy_img=False)
        only_other_3.filter_by_tags(need_tag = model_xjDP_kkx.labeles_checkedOut)
        only_k.filter_by_tags(need_tag=['K'])
        dete_res_kkx.filter_by_tags(remove_tag=['K'])
        for each_dete_obj in only_k:
            is_in = False
            for each_dete_obj_2 in only_other_3:
                each_iou = ResTools.polygon_iou_1(each_dete_obj.get_points(),each_dete_obj_2.get_points()) ##
                #print('--other* iou-->{} ,other*:{}, K: {}'.format(each_iou,each_dete_obj_2.get_points(),each_dete_obj.get_points()))
                if each_iou > 0.8:
                    is_in = True
            if not is_in:
                dete_res_kkx.add_obj_2(each_dete_obj)

        # --------------------------------------------------------------------------------------------------------------
        # xjDP_cls
        dete_res_kkx.filter_by_tags(need_tag=['K', 'KG', 'K2', 'KG2'])

        #
        for each_dete_obj in dete_res_kkx:
            # dete
            im = dete_res_kkx.get_sub_img_by_dete_obj_new(each_dete_obj)
            label, prob = model_xjDP_cls.detect(im, 'resizedName')

            each_dete_obj.conf = float(prob)
            each_dete_obj.des = each_dete_obj.tag

            if label == 1 and prob > model_xjDP_cls.confThresh:
                each_dete_obj.tag = 'dp_missed'
            elif label == 2:
                each_dete_obj.tag = 'kkgLm'
            else:
                if prob > model_xjDP_cls.confThresh:
                    each_dete_obj.tag = 'dp_normal'
                else:
                    each_dete_obj.tag = 'dp_missed'
                    each_dete_obj.conf = 1.01

        # filter K, KG, Lm
        dete_res_kkx.filter_by_tags(need_tag=['dp_missed', 'dp_normal','kkgLm'])
        #
        dete_res_ljc += dete_res_kkx
        #
        return dete_res_ljc

    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)


