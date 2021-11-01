# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os, sys

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')
sys.path.insert(0, lib_path)
import argparse
import cv2
import shutil
import json
import torch
import numpy as np
import threading
from PIL import Image
import uuid
import time
from lib.detect_libs.yolov5Detection import YOLOV5Detection
from lib.detect_utils.timer import Timer
from lib.detect_libs.fasterDetectionPyTorch import FasterDetectionPytorch
from lib.detect_libs.vggClassify import VggClassify
from lib.detect_libs.clsDetectionPyTorch import ClsDetectionPyTorch
from lib.detect_libs.ljcY5Detection import LjcDetection
from lib.detect_libs.kkgY5Detection import KkgDetection
from lib.detect_libs.clsDetectionPyTorch import ClsDetectionPyTorch
#
from lib_xjQX.detect_libs.ljjxjR2cnnDetection import ljcR2cnnDetection
from lib.detect_libs.xjdectR2cnnPytorchDetection import XjdectR2cnnDetection
from lib_xjQX.detect_libs.xjDeeplabDetection import xjDeeplabDetection
#
from lib.JoTools.txkjRes.resTools import ResTools
from lib.JoTools.utils.FileOperationUtil import FileOperationUtil
from lib.JoTools.utils.CsvUtil import CsvUtil
from lib.JoTools.txkjRes.deteRes import DeteRes
from lib.JoTools.txkjRes.deteObj import DeteObj
from lib.JoTools.txkjRes.deteAngleObj import DeteAngleObj
from lib.JoTools.utils.JsonUtil import JsonUtil

# kkxTC vit
from lib.detect_libs.clsViTDetection import ClsViTDetection

# jyhQX
from lib.detect_libs.r2cnnPytorchDetection import R2cnnDetection
from lib.detect_libs.jyhDeeplabDetection import jyhDeeplabDetection
import judge_angle_fun
#
from fangtian_info_dict import M_dict, M_model_list, key_M_dict, tag_code_dict


# fixme 如何告诉外界，当前的模型处于三种状态中的哪一种（init，running，end）


# fixme 将读取图片全部改为传入矩阵的方式进行，
# fixme 使用 JoTools 函数


def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    #
    parser.add_argument('--imgDir', dest='imgDir', type=str, default=r"/usr/input_picture")
    parser.add_argument('--modelList', dest='modelList', default="M1,M2,M3,M4,M5,M6,M7,M8,M9")
    parser.add_argument('--jsonPath', dest='jsonPath', default=r"/usr/input_picture_attach/pictureName.json")
    parser.add_argument('--outputDir', dest='outputDir', default=r"/usr/output_dir")
    parser.add_argument('--signDir', dest='signDir', default=r"/usr/sign")
    #
    parser.add_argument('--scriptIndex', dest='scriptIndex', default=r"1-1")
    #
    parser.add_argument('--gpuID', dest='gpuID', type=int, default=0)
    parser.add_argument('--port', dest='port', type=int, default=45452)
    parser.add_argument('--gpuRatio', dest='gpuRatio', type=float, default=0.3)
    parser.add_argument('--host', dest='host', type=str, default='127.0.0.1')
    parser.add_argument('--logID', dest='logID', type=str, default=str(uuid.uuid1())[:6])
    parser.add_argument('--objName', dest='objName', type=str, default='')
    #
    args = parser.parse_args()
    return args


def get_json_dict(json_path):
    img_name_json_dict = {}
    #
    name_info = JsonUtil.load_data_from_json_file(json_path)
    for each in name_info:
        img_name_json_dict[each["fileName"]] = each["originFileName"]
    return img_name_json_dict


def get_img_path_list_from_sign_dir(sign_txt_path, img_dir):
    """从 sign_dir 中读取需要处理的文件"""

    # fixme 每一个进程分配使用不一样的文件夹，根据进程的 index 决定使用哪一个文件夹，很少的文件可以通过减小 batch size 来进行调节

    # fixme batch 参数可以初始化的时候进行传入
    batch_size = 50

    if not os.path.exists(sign_txt_path):
        print("* sign txt dir not exists : {0}".format(sign_txt_path))
        return [], None

    img_dir_list = []
    with open(sign_txt_path, 'r') as sign_txt_file:
        for each_line in sign_txt_file:
            each_img_dir = os.path.join(img_dir, each_line.strip())
            img_dir_list.append(each_img_dir)
    #
    for each_index in range(script_index - 1, len(img_dir_list), script_num):
        each_img_dir = img_dir_list[each_index]
        if os.path.exists(each_img_dir):
            img_path_list = list(
                FileOperationUtil.re_all_file(each_img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG']))
            if len(img_path_list) != 0:
                return img_path_list, each_img_dir
    return [], None


def screen(y, img):
    # screen brightness
    _, _, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    vmedian = np.median(v)
    if vmedian < 35:
        y = '0'
    # screen obscure
    blurry = cv2.Laplacian(img, cv2.CV_64F).var()
    if blurry < 200:
        y = '0'
    return y


def get_model_list_from_img_name(img_name, M_list):
    """从文件名中获取 model_list，传入的是文件名不是完整的路径"""
    model_set = set()
    all_model_set = set()
    for key in M_list:
        for each_model_name in M_model_list[key]:
            all_model_set.add(each_model_name)

    is_empty = True
    for each_key in key_M_dict:
        # for each_key in key_M_dict:
        if each_key in img_name:
            is_empty = False
            # if key_M_dict[each_key] in M_list:
            for each_model_name in key_M_dict[each_key]:
                if each_model_name in all_model_set:
                    model_set.add(each_model_name)
    if len(model_set) > 0:
        return list(model_set)
    elif is_empty:
        return all_model_set
    else:
        return model_set


def model_restore(args, scriptName, model_list=None):
    """模型预热"""

    model_dict = {}

    if model_list is None:
        model_list = ['nc', 'jyzZB', 'fzc', 'fzcRust', 'ljcRust', 'fncDK', 'kkxTC', 'kkxQuiting', 'kkxRust', 'waipo',
                      'xjQX', 'jyhQX']

    if "xjQX" in model_list:
        # model_xjQX_1 = ljcR2cnnDetection(args, "ljjxj", scriptName)
        # model_xjQX_1.model_restore()
        model_xjQX_1 = XjdectR2cnnDetection(args, "xjQX_ljc", scriptName)
        model_xjQX_1.model_restore()

        #
        model_xjQX_2 = xjDeeplabDetection(args, "xj_deeplab", scriptName)
        model_xjQX_2.model_restore()
        model_dict["model_xjQX_1"] = model_xjQX_1
        model_dict["model_xjQX_2"] = model_xjQX_2

    if "jyzZB" in model_list:
        model_jyzZB_1 = YOLOV5Detection(args, "jyz", scriptName)
        model_jyzZB_1.model_restore()
        model_jyzZB_2 = YOLOV5Detection(args, "jyzzb", scriptName)
        model_jyzZB_2.model_restore()
        model_dict["model_jyzZB_1"] = model_jyzZB_1
        model_dict["model_jyzZB_2"] = model_jyzZB_2

    if "nc" in model_list:
        model_nc = YOLOV5Detection(args, "nc", scriptName)
        model_nc.model_restore()
        model_dict["model_nc"] = model_nc

    if "fzc" in model_list:
        model_fzc_1 = FasterDetectionPytorch(args, "fzc_step_one", scriptName)
        model_fzc_1.model_restore()
        model_fzc_2 = VggClassify(args, "fzc_step_new", scriptName)
        model_fzc_2.model_restore()
        model_dict["model_fzc_1"] = model_fzc_1
        model_dict["model_fzc_2"] = model_fzc_2

    if "fzcRust" in model_list:
        model_fzc_rust = ClsDetectionPyTorch(args, "fzc_rust", scriptName)
        model_fzc_rust.model_restore()
        model_dict["model_fzc_rust"] = model_fzc_rust

    if "fncDK" in model_list:
        model_fnc = YOLOV5Detection(args, "fnc", scriptName)
        model_fnc.model_restore()
        model_dict["model_fnc"] = model_fnc

    if "kkxTC" in model_list or "kkxQuiting" in model_list or "kkxRust" in model_list:
        model_kkxTC_1 = LjcDetection(args, "kkxTC_ljc", scriptName)
        model_kkxTC_1.model_restore()
        model_kkxTC_2 = KkgDetection(args, "kkxTC_kkx", scriptName)
        model_kkxTC_2.model_restore()
        # vit
        # model_kkxTC_3 = ClsDetectionPyTorch(args, "kkxTC_lm_cls", scriptName)
        model_kkxTC_3 = ClsViTDetection(args, "kkxTC_lm_cls_vit", scriptName)
        model_kkxTC_3.model_restore()
        #
        model_kkxQuiting = ClsDetectionPyTorch(args, "kkxQuiting_cls", scriptName)
        model_kkxQuiting.model_restore()
        model_kkxRust = VggClassify(args, "kkxRust", scriptName)
        model_kkxRust.model_restore()
        model_dict["model_kkxTC_1"] = model_kkxTC_1
        model_dict["model_kkxTC_2"] = model_kkxTC_2
        model_dict["model_kkxTC_3"] = model_kkxTC_3
        model_dict["model_kkxQuiting"] = model_kkxQuiting
        model_dict["model_kkxRust"] = model_kkxRust

    if "waipo" in model_list:
        model_waipo = YOLOV5Detection(args, "waipo", scriptName)
        model_waipo.model_restore()
        model_dict["model_waipo"] = model_waipo

    if "ljcRust" in model_list:
        model_ljc_rust_1 = YOLOV5Detection(args, "ljc_rust_one", scriptName)
        model_ljc_rust_1.model_restore()
        model_ljc_rust_2 = ClsDetectionPyTorch(args, "ljc_rust_two", scriptName)
        model_ljc_rust_2.model_restore()
        model_dict["model_ljc_rust_1"] = model_ljc_rust_1
        model_dict["model_ljc_rust_2"] = model_ljc_rust_2

    if "jyhQX" in model_list:
        model_jyhQX_1 = YOLOV5Detection(args, "jyhqx_one", scriptName)
        model_jyhQX_1.model_restore()
        model_jyhQX_2 = R2cnnDetection(args, "jyhqx_two", scriptName)
        model_jyhQX_2.model_restore()
        model_jyhQX_3 = jyhDeeplabDetection(args, "jyhqx_three", scriptName)
        model_jyhQX_3.model_restore()
        model_dict["model_jyhqx_1"] = model_jyhQX_1
        model_dict["model_jyhqx_2"] = model_jyhQX_2
        model_dict["model_jyhqx_3"] = model_jyhQX_3

    return model_dict


def model_dete(img_path, model_dict, model_list=None):
    """进行模型检测"""

    name = os.path.split(img_path)[1]

    if model_list is None:
        model_list = ['nc', 'jyzZB', 'fzc', 'fzcRust', 'ljcRust', 'fncDK', 'kkxTC', 'kkxQuiting', 'kkxRust', 'waipo',
                      'xjQX', 'jyhQX']

    # im
    data = {"path": img_path}
    im = np.array(Image.open(data['path']))

    # dete result for all
    dete_res_all = DeteRes()
    dete_res_all.img_path = img_path

    if "jyzZB" in model_list:

        if ("model_jyzZB_1" not in model_dict):
            print("* error : no model : model_jyzZB_1")

        if ("model_jyzZB_2" not in model_dict):
            print("* error : no model : model_jyzZB_2")

        try:
            model_jyzZB_1 = model_dict["model_jyzZB_1"]
            model_jyzZB_2 = model_dict["model_jyzZB_2"]

            # jyzZB step_1
            dete_res_jyzZB = model_jyzZB_1.detectSOUT(path=data['path'], image_name=name)
            # torch.cuda.empty_cache()
            # ---------------------------------------
            # jyzZB step_2
            result_res = DeteRes(assign_img_path=data['path'])
            result_res.img_path = data['path']
            #
            result_res.file_name = name
            for each_dete_obj in dete_res_jyzZB:
                each_dete_obj.do_augment([150, 150, 150, 150], dete_res_jyzZB.width, dete_res_jyzZB.height,
                                         is_relative=False)
                each_im = dete_res_jyzZB.get_sub_img_by_dete_obj(each_dete_obj)
                new_dete_res = model_jyzZB_2.detectSOUT(image=each_im, image_name=each_dete_obj.get_name_str())
                new_dete_res.offset(each_dete_obj.x1, each_dete_obj.y1)
                result_res += new_dete_res

            # new logic
            result_res.filter_tag1_by_tag2_with_nms(['jyzSingle'], ['jyzhead'], 0.5)

            # MYLOG.info(result_res.get_fzc_format())
            result_res.do_nms_center_point(ignore_tag=True)
            result_res.update_tags({"jyzSingle": "jyzzb"})
            dete_res_all += result_res
            # torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_frame.f_globals["__file__"])
            print(e.__traceback__.tb_lineno)

    if "nc" in model_list:

        if ("model_nc" not in model_dict):
            print("* error : no model : model_nc")

        try:

            model_nc = model_dict["model_nc"]

            nc_dete_res = model_nc.detectSOUT(path=data['path'], image_name=name)
            dete_res_all += nc_dete_res
            # torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_frame.f_globals["__file__"])
            print(e.__traceback__.tb_lineno)

    if "fzc" in model_list:

        if ("model_fzc_1" not in model_dict):
            print("* error : no model : model_fzc_1")

        if ("model_fzc_2" not in model_dict):
            print("* error : no model : model_fzc_2")

        if ("model_fzc_rust" not in model_dict):
            # print("* error : no model : model_fzc_rust")
            pass

        try:

            if "model_fzc_1" in model_dict or "model_fzc_2" in model_dict:
                model_fzc_1 = model_dict["model_fzc_1"]
                model_fzc_2 = model_dict["model_fzc_2"]

            if "model_fzc_rust" in model_dict:
                model_fzc_rust = model_dict["model_fzc_rust"]

            # step_1
            dete_res_fzc = model_fzc_1.detectSOUT(path=data['path'], image_name=name)
            # step_2
            for each_dete_obj in dete_res_fzc:
                crop_array = dete_res_fzc.get_sub_img_by_dete_obj(each_dete_obj, RGB=False,
                                                                  augment_parameter=[0.1, 0.1, 0.1, 0.1])
                new_label, conf = model_fzc_2.detect_new(crop_array, name)
                #
                each_dete_obj.tag = new_label
                each_dete_obj.conf = conf

                # for rust
                # if each_dete_obj.tag in ['', '', '']:

                #
                if each_dete_obj.tag == "fzc_broken":
                    if each_dete_obj.conf > 0.9:
                        each_dete_obj.tag = "fzc_broken"
                    else:
                        each_dete_obj.tag = "other_fzc_broken"
                elif each_dete_obj.tag == "other":
                    each_dete_obj.tag = "other_other"
                else:
                    if each_dete_obj.conf > 0.6:
                        each_dete_obj.tag = "Fnormal"
                    else:
                        each_dete_obj.tag = "other_Fnormal"
                #
                dete_res_all.add_obj_2(each_dete_obj)

                if "model_fzc_rust" in model_dict:
                    # rust
                    if new_label in ["yt", "zd_yt"]:
                        crop_array_rust = dete_res_fzc.get_sub_img_by_dete_obj(each_dete_obj, RGB=False)
                        rust_index, rust_f = model_fzc_rust.detect(crop_array_rust)
                        rust_label = ["fzc_normal", "fzc_rust"][int(rust_index)]
                        rust_f = float(rust_f)
                        #
                        each_dete_rust = each_dete_obj.deep_copy()
                        each_dete_rust.tag = rust_label
                        #
                        dete_res_all.add_obj_2(each_dete_rust)
            # torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_frame.f_globals["__file__"])
            print(e.__traceback__.tb_lineno)

    if "kkxTC" in model_list or "kkxQuiting" in model_list or "kkxRust" in model_list:

        if ("model_kkxTC_1" not in model_dict):
            print("* error : no model : model_kkxTC_1")

        if ("model_kkxTC_2" not in model_dict):
            print("* error : no model : model_kkxTC_2")

        if ("model_kkxTC_3" not in model_dict):
            print("* error : no model : model_kkxTC_3")

        if ("model_kkxRust" not in model_dict):
            print("* error : no model : model_kkxRust")

        if ("model_kkxQuiting" not in model_dict):
            print("* error : no model : model_kkxQuiting")

        try:

            # if "model_kkxTC_1" in model_dict or "model_kkxTC_2" in model_dict or "model_kkxTC_3" in model_dict:
            model_kkxTC_1 = model_dict["model_kkxTC_1"]
            model_kkxTC_2 = model_dict["model_kkxTC_2"]
            model_kkxTC_3 = model_dict["model_kkxTC_3"]

            if "model_kkxRust" in model_dict:
                model_kkxRust = model_dict["model_kkxRust"]

            if "model_kkxQuiting" in model_dict:
                model_kkxQuiting = model_dict["model_kkxQuiting"]

            # kkxTC_1
            kkxTC_1_out = model_kkxTC_1.detect(im, name)
            if len(kkxTC_1_out[0]) > 0:
                voc_labels = model_kkxTC_1.post_process(*kkxTC_1_out)
                # MYLOG.info("detect result:", voc_labels)
                kkxTC_1_results = model_kkxTC_1.postProcess2(im, *kkxTC_1_out)
            else:
                kkxTC_1_results = []
                #
            kkxTC_1_dete_res = DeteRes()
            kkxTC_1_dete_res.img_path = data['path']
            for i, each_res in enumerate(kkxTC_1_results):
                label, score, [xmin, ymin, xmax, ymax] = each_res
                ljc_resizedName = name + '_' + label + '_' + str(i) + '.jpg'
                # add up_right obj
                kkxTC_1_dete_res.add_obj(int(xmin), int(ymin), int(xmax), int(ymax), str(label), conf=-1, assign_id=i,
                                         describe=ljc_resizedName)
            #
            kkxTC_1_dete_res.do_nms(0.3)
            kkxTC_1_save_dir = model_kkxTC_1.resizedImgPath
            kkxTC_1_dete_res.crop_dete_obj(kkxTC_1_save_dir)
            # ---------------------------------------
            # kkxTC_2
            ###  单连接件 ###
            kkxTC_2_dete_kg_lm = kkxTC_1_dete_res.deep_copy(copy_img=False)
            kkxTC_2_dete_kg_lm.reset_alarms([])

            # 遍历每一个连接件正框
            for each_dete_obj in kkxTC_1_dete_res.alarms:
                each_dete_kg_lm = kkxTC_1_dete_res.deep_copy(copy_img=False)
                each_dete_kg_lm.reset_alarms([])
                # get array 连接件正框图片矩阵 np.array
                # each_sub_array = kkxTC_1_dete_res.get_sub_img_by_dete_obj(each_dete_obj,RGB=True)
                each_sub_array = kkxTC_1_dete_res.get_sub_img_by_dete_obj_from_crop(each_dete_obj, RGB=False)
                # 小金具定位检测结果集合 on a ljc martrix-cap
                kkxTC_2_out = model_kkxTC_2.detect(each_sub_array, name)
                if len(kkxTC_2_out[0]) > 0:
                    voc_labels = model_kkxTC_2.post_process(*kkxTC_2_out)
                    ## 过滤最小尺寸 ##
                    voc_labels = model_kkxTC_2.checkDetectBoxAreas(voc_labels)

                    for each_obj in voc_labels:
                        ## label, i, xmin, ymin, xmax, ymax,p
                        new_dete_obj = DeteObj(each_obj[2], each_obj[3], each_obj[4], each_obj[5], tag=each_obj[0],
                                               conf=float(each_obj[6]), assign_id=each_dete_obj.id)
                        each_dete_kg_lm.add_obj_2(new_dete_obj)

                    ## +xmap +ymap 坐标还原至原图
                    each_dete_kg_lm.offset(each_dete_obj.x1, each_dete_obj.y1)
                    # merge
                    kkxTC_2_dete_kg_lm += each_dete_kg_lm

            # 业务逻辑：other* 和 dense内的K过滤
            kkxTC_2_dete_res = kkxTC_2_dete_kg_lm.deep_copy(copy_img=False)
            only_other_3 = kkxTC_2_dete_kg_lm.deep_copy(copy_img=False)
            only_k = kkxTC_2_dete_kg_lm.deep_copy(copy_img=False)
            only_other_3.filter_by_tags(need_tag=model_kkxTC_2.labeles_checkedOut)
            only_k.filter_by_tags(need_tag=['K'])
            kkxTC_2_dete_res.filter_by_tags(remove_tag=['K'])
            #
            for each_dete_obj in only_k:
                is_in = False
                for each_dete_obj_2 in only_other_3:
                    each_iou = ResTools.polygon_iou_1(each_dete_obj.get_points(), each_dete_obj_2.get_points())  ##
                    # print('--other* iou-->{} ,other*:{}, K: {}'.format(each_iou,each_dete_obj_2.get_points(),each_dete_obj.get_points()))
                    if each_iou > 0.8:
                        is_in = True
                if not is_in:
                    kkxTC_2_dete_res.add_obj_2(each_dete_obj)
            #
            kkxTC_2_dete_res.do_nms(0.3)
            # 删除裁剪的小图
            kkxTC_1_dete_res.del_sub_img_from_crop()

            # ---------------------------------------

            # kkxTC_3 | kkxQuiting | kkxRust
            kkxTC_dete_res = DeteRes()
            kkxQuiting_dete_res = DeteRes()
            kkxRust_dete_res = DeteRes()
            #
            for each_dete_obj in kkxTC_2_dete_res:
                if each_dete_obj.tag not in ['K', 'KG', 'Lm', 'K2', 'KG2']:
                    continue
                #
                each_im = kkxTC_2_dete_res.get_sub_img_by_dete_obj(each_dete_obj)
                # -----------------
                # kkxTC
                # if "kkxTC" in model_list:

                # print("* ", label, prob)

                label, prob = model_kkxTC_3.detect(each_im, 'resizedName')
                label = str(label)
                each_dete_obj.conf = float(prob)
                each_dete_obj.des = each_dete_obj.tag

                # print("* ", label, prob, model_kkxTC_3.confThresh, type(label), type(each_dete_obj.tag))

                if label == '2' or each_dete_obj.tag == 'Lm':
                    each_dete_obj.tag = 'Lm'

                elif label == '1' and prob > model_kkxTC_3.confThresh:
                    each_dete_obj.tag = 'K'
                else:
                    each_dete_obj.tag = 'Xnormal'
                kkxTC_dete_res.add_obj_2(each_dete_obj)
                # -----------------
                # kkxRust
                if "model_kkxRust" in model_dict:
                    if "kkxRust" in model_list:
                        new_label, conf = model_kkxRust.detect_new(each_im, name)
                        new_dete_obj_rust = each_dete_obj.deep_copy()
                        if new_label == 'kkx_rust' and conf > 0.8:
                            if each_dete_obj.tag in ["Lm"]:
                                new_dete_obj_rust.tag = 'Lm_rust'
                            else:
                                new_dete_obj_rust.tag = 'K_KG_rust'
                        else:
                            new_dete_obj_rust.tag = 'Lnormal'
                        kkxRust_dete_res.add_obj_2(new_dete_obj_rust)
                # -----------------
                # kkxQuiting
                if "model_kkxQuiting" in model_dict:
                    if "kkxQuiting" in model_list:
                        # 0:销脚可见 1:退出 2:销头销脚正对
                        if each_dete_obj.tag in ["Xnormal"]:
                            label, prob = model_kkxQuiting.detect(each_im, 'resizedName')
                            if label == '1' and prob > 0.5:
                                new_dete_obj = each_dete_obj.deep_copy()
                                new_dete_obj.tag = 'kkxTC'
                                kkxQuiting_dete_res.add_obj_2(new_dete_obj)

            if "kkxTC" in model_list:
                dete_res_all += kkxTC_dete_res

            if "kkxQuiting" in model_list:
                dete_res_all += kkxQuiting_dete_res

            if "kkxRust" in model_list:
                dete_res_all += kkxRust_dete_res

            # torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_frame.f_globals["__file__"])
            print(e.__traceback__.tb_lineno)

    """

    if "xjQX" in model_list:

        if ("model_xjQX_1" not in model_dict):
            print("* error : no model : model_xjQX_1")

        if ("model_xjQX_2" not in model_dict):
            print("* error : no model : model_xjQX_2")

        try:

            model_xjQX_1 = model_dict["model_xjQX_1"]
            model_xjQX_2 = model_dict["model_xjQX_2"]

            xjQX_dete_res = DeteRes()
            #xjQX_dete_res.img_path = data['path']

            detectBoxes = model_xjQX_1.detect(im, name)
            results = model_xjQX_1.postProcess2(im, name, detectBoxes)

            print('*'*50)
            print(detectBoxes)
            print(results)
            print('*'*50)

            # 
            for xjBox in results:
                resizedName = xjBox['resizedName']
                resizedImg = im[xjBox['ymin']:xjBox['ymax'],xjBox['xmin']:xjBox['xmax']]
                segImage = model_xjQX_2.detect(resizedImg,resizedName)
                result = model_xjQX_2.postProcess(segImage,resizedName,xjBox)

                # add obj
                if "position" in result:
                    x1, y1, w, h = result["position"]
                    x2, y2 = x1 + w, y1 + h
                    tag = result["class"]
                    xjQX_dete_res.add_obj(x1, y1, x2, y2, tag, conf=-1, assign_id=-1, describe='')

            dete_res_all += xjQX_dete_res
            #torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_frame.f_globals["__file__"])
            print(e.__traceback__.tb_lineno)

    """

    if "jyhQX" in model_list:

        try:

            model_jyhqx_1 = model_dict["model_jyhqx_1"]
            model_jyhqx_2 = model_dict["model_jyhqx_2"]
            model_jyhqx_3 = model_dict["model_jyhqx_3"]

            # ----------------------------------------------------------------------------------------------------------
            # dete res
            jyhqx_1_dete_res = model_jyhqx_1.detectSOUT(path=data['path'], image_name=name)
            jyhqx_1_dete_res.img_path = data['path']
            #

            jyhqx_1_dete_res.filter_by_tags(["fhjyz", "upring", "downring"])

            # 将 fhjyz 按照一定的规则进行扩展
            for each_obj in jyhqx_1_dete_res:
                if each_obj.tag == "fhjyz":
                    each_obj.do_augment([50, 50, 50, 50], jyhqx_1_dete_res.width, jyhqx_1_dete_res.height,
                                        is_relative=False)

            # 剔除那些均压环和绝缘子无交集的部分
            dete_res_jyz = jyhqx_1_dete_res.deep_copy(copy_img=False)
            dete_res_jyz.filter_by_tags(["fhjyz"])
            #
            dete_res_jyh = jyhqx_1_dete_res.deep_copy(copy_img=False)
            dete_res_jyh.filter_by_tags(["upring", "downring"])

            # 判断绝缘子和均压环之间的交集
            for each_jyh in dete_res_jyh:
                each_dete_res_jyz = dete_res_jyz.deep_copy(copy_img=False)
                each_dete_res_jyz.filter_by_mask(each_jyh.get_points(), need_in=True, cover_index_th=0.0001)
                #
                if len(each_dete_res_jyz) < 2:
                    jyhqx_1_dete_res.del_dete_obj(each_jyh)

            # 去除其中的绝缘子
            jyhqx_1_dete_res.filter_by_tags(["upring", "downring"])

            for each_obj in jyhqx_1_dete_res:
                if each_obj.tag == "downring":
                    each_obj.do_augment([0, 0, 1, 0], jyhqx_1_dete_res.width, jyhqx_1_dete_res.height, is_relative=True)
                elif each_obj.tag == "upring":
                    each_obj.do_augment([0, 0, 0, 1], jyhqx_1_dete_res.width, jyhqx_1_dete_res.height, is_relative=True)

            # ----------------------------------------------------------------------------------------------------------

            for each_dete_obj in jyhqx_1_dete_res:
                each_im = jyhqx_1_dete_res.get_sub_img_by_dete_obj(each_dete_obj)
                a = model_jyhqx_2.detect(each_im, name)
                #
                if len(a) < 1:
                    # dete_res.del_dete_obj(each_dete_obj)
                    # 在最后一步根据 des 进行过滤就行了，不用在这边删除
                    pass
                else:
                    point_0, point_1, point_2, point_3, _ = a[0]
                    x1, y1 = int((point_0[0] + point_1[0]) / 2), int((point_0[1] + point_1[1]) / 2)
                    x2, y2 = int((point_2[0] + point_3[0]) / 2), int((point_2[1] + point_3[1]) / 2)
                    #
                    each_dete_obj.des = "[{0},{1},{2},{3}]".format(x1, y1, x2, y2)
            # ----------------------------------------------------------------------------------------------------------

            # itoration
            for each_dete_obj in jyhqx_1_dete_res:
                # do filter
                if each_dete_obj.tag not in ['upring', 'downring'] or each_dete_obj.des in [None, "", " "]:
                    continue
                # do dete
                im = jyhqx_1_dete_res.get_sub_img_by_dete_obj(each_dete_obj, RGB=False)
                # print(dete_res_3.width,dete_res_3.height)
                x_add, y_add = each_dete_obj.x1, each_dete_obj.y1
                seg_image = model_jyhqx_3.detect(im, 'resizedName')
                result = model_jyhqx_3.postProcess(seg_image, 'resizedName')

                # fixme 看一下这一步是不是需要
                if not result:
                    continue

                # get_angle_1
                angle_deep_lab = result['angle']
                line_jyh = [result['xVstart'], result['yVstart'], result['xHend'], result['yHend']]
                line_jyz_1 = [result['xVstart'], result['yVstart'], result['xVend'], result['yVend']]
                # get_angle_2
                des = each_dete_obj.des
                line_jyz_2 = eval(des)
                angle_r2cnn = judge_angle_fun.angle_r2cnn(x_add, y_add, each_dete_obj, result, des)
                # judge Hnormal or fail
                Hnormal_or_fail, line_index, use_angle = judge_angle_fun.Judge_Hnormal_fail(angle_deep_lab, angle_r2cnn)
                each_dete_obj.tag = Hnormal_or_fail
                # get lines to draw
                lines_to_draw = judge_angle_fun.get_lines_to_draw(line_jyh, line_jyz_1, line_jyz_2, line_index, x_add,
                                                                  y_add, use_angle)
                each_dete_obj.des = str(lines_to_draw)

            dete_res_all += jyhqx_1_dete_res

            # ----------------------------------------------------------------------------------------------------------
        except Exception as e:
            print("error")
            print(e)
            print(e.__traceback__.tb_frame.f_globals["__file__"])
            print(e.__traceback__.tb_lineno)

    # set confidence as 1 when confidence less then 0
    for each_dete_obj in dete_res_all:
        if each_dete_obj.conf < 0:
            each_dete_obj.conf = 1

    # nms
    dete_res_all.do_nms(0.3)

    # filter by tags
    dete_res_all.filter_by_tags(need_tag=list(tag_code_dict.keys()))

    # update tags
    dete_res_all.update_tags(tag_code_dict)

    # save xml
    dete_res_all.print_as_fzc_format()
    save_dir = os.path.join(output_dir, "save_res")
    os.makedirs(save_dir, exist_ok=True)
    each_save_name = os.path.split(img_path)[1]
    each_save_path = os.path.join(save_dir, each_save_name)
    each_save_path_xml = os.path.join(save_dir, each_save_name[:-4] + '.xml')
    # dete_res_all.draw_dete_res(each_save_path)
    dete_res_all.save_to_xml(each_save_path_xml)

    # empty cache
    torch.cuda.empty_cache()

    return dete_res_all


if __name__ == '__main__':

    args = parse_args()

    img_dir = args.imgDir.strip()
    json_path = args.jsonPath
    output_dir = args.outputDir.strip()
    sign_dir = args.signDir.strip()
    log_path = os.path.join(output_dir, "log")
    csv_path = os.path.join(output_dir, "result.csv")
    sign_txt_path = os.path.join(sign_dir, "img_dir_to_dete.txt")
    sign_error_dir = os.path.join(sign_dir, "dete_error_img")
    os.makedirs(sign_error_dir, exist_ok=True)
    #
    script_num, script_index = args.scriptIndex.strip().split('-')
    script_num, script_index = int(script_num), int(script_index)
    # ---------------------------
    print('-' * 50)
    print("* {0} : {1}".format("img_dir", img_dir))
    print("* {0} : {1}".format("json_path", json_path))
    print("* {0} : {1}".format("log_path", log_path))
    print("* {0} : {1}".format("csv_path", csv_path))
    print("* {0} : {1}".format("dete_eror_dir", sign_error_dir))
    print("* script_num-script_index : {0}-{1}".format(script_num, script_index))
    print('-' * 50)

    dete_img_index = 0

    # model_list
    assign_model_list = args.modelList.strip().split(',')

    # warm up
    print("* start warm model ")
    scriptName = os.path.basename(__file__).split('.')[0]
    #
    all_model_list = ['nc', 'jyzZB', 'fzc', 'fzcRust', 'kkxTC', 'kkxQuiting', 'xjQX', 'jyhQX']

    model_dict = model_restore(args, scriptName, all_model_list)
    print("* warm model success ")

    start_time = time.time()

    while True:
        img_path_list, each_img_dir = get_img_path_list_from_sign_dir(sign_txt_path, img_dir)
        # dete
        for each_img_path in img_path_list:
            dete_img_index += 1
            print("* {0} : {1}".format(dete_img_index, each_img_path))
            try:
                # over time continue
                # if time.time() - start_time < max_use_time:
                each_model_list = get_model_list_from_img_name("", assign_model_list)
                try:
                    each_dete_res = model_dete(each_img_path, model_dict, each_model_list)
                except Exception as e:
                    print(e)
                    print(e.__traceback__.tb_frame.f_globals["__file__"])
                    print(e.__traceback__.tb_lineno)
                    # move error img to assign dir
                    shutil.move(each_img_path, os.path.join(sign_error_dir, os.path.split(each_img_path)[1]))
                #
                if os.path.exists(each_img_path):
                    os.remove(each_img_path)
            except Exception as e:
                print(e)
                print(e.__traceback__.tb_frame.f_globals["__file__"])
                print(e.__traceback__.tb_lineno)
            #
            print('-'*50)

        # when dete finished , delete the img dir
        if each_img_dir is not None:
            if os.path.exists(each_img_dir):
                if len(list(FileOperationUtil.re_all_file(each_img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG']))) == 0:
                    os.rmdir(each_img_dir)
        #
        time.sleep(2)


