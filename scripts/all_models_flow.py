# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os, sys

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')
sys.path.insert(0, lib_path)
#
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
import copy
#
from fangtian_info_dict import M_dict, M_model_list, key_M_dict, tag_code_dict
# time analysis
from lib.JoTools.utils.FileOperationUtil import FileOperationUtil
from lib.JoTools.txkjRes.deteRes import DeteRes
from lib.JoTools.txkjRes.deteObj import DeteObj
from lib.JoTools.utils.DecoratorUtil import DecoratorUtil
# load model dete script
from dete import dete_nc, dete_fzc, dete_kkx, dete_xjQX, dete_jyhQX, dete_jyzZB, all_model_restore

# ------------------ del -----------------------------------------
# todo status
cpu_num = 1                                                 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
# ------------------ del -----------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    #
    parser.add_argument('--imgDir', dest='imgDir', type=str, default=r"/usr/input_picture")
    parser.add_argument('--modelList', dest='modelList', default="M1,M2,M3,M4,M5,M6,M7,M8,M9")
    parser.add_argument('--jsonPath', dest='jsonPath', default=r"/usr/input_picture_attach/pictureName.json")
    parser.add_argument('--outputDir', dest='outputDir', default=r"/usr/output_dir")
    parser.add_argument('--signDir', dest='signDir', default=r"/v0.0.1/sign")
    #
    parser.add_argument('--scriptIndex', dest='scriptIndex', default=r"1-1")
    parser.add_argument('--deteMode', dest='deteMode',type=int, default=0)            # 0 : 处理每个文件夹中的 n 分之一，1：处理 n 分之一的文件夹
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

def get_img_path_list_from_sign_dir(sign_txt_path, img_dir, dete_mode):
    """从 sign_dir 中读取需要处理的文件"""

    if not os.path.exists(sign_txt_path):
        print("* sign txt dir not exists : {0}".format(sign_txt_path))
        return [], None

    img_dir_list = []
    with open(sign_txt_path, 'r') as sign_txt_file:
        for each_line in sign_txt_file:
            each_img_dir = os.path.join(img_dir, each_line.strip())
            img_dir_list.append(each_img_dir)
    #
    if dete_mode == 1:
        for each_index in range(script_index - 1, len(img_dir_list), script_num):
            each_img_dir = img_dir_list[each_index]
            if os.path.exists(each_img_dir):
                img_path_list = list(
                    FileOperationUtil.re_all_file(each_img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG']))
                if len(img_path_list) != 0:
                    return img_path_list, each_img_dir
    elif dete_mode == 0:
        for each_img_dir in img_dir_list:
            if os.path.exists(each_img_dir):
                res_list = []
                img_path_list = list(FileOperationUtil.re_all_file(each_img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG']))
                for each_img_index in range(script_index -1 , len(img_path_list), script_num):
                    res_list.append(img_path_list[each_img_index])
                #
                if len(res_list) != 0:
                    return res_list, each_img_dir
    else:
        raise ValueError("error : dete mode can only in [0, 1]")
    return [], None

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

@DecoratorUtil.time_this
def model_dete(img_path, model_dict, model_list):
    """进行模型检测"""

    name = os.path.split(img_path)[1]

    # im
    im = np.array(Image.open(img_path))
    data = {"path": img_path, 'name': name, 'im':im}
    dete_res_all = DeteRes()
    dete_res_all.img_path = img_path

    #
    save_dir = os.path.join(output_dir, "save_res")
    os.makedirs(save_dir, exist_ok=True)
    each_save_name = os.path.split(img_path)[1]
    each_save_path_xml = os.path.join(save_dir, each_save_name[:-4] + '.xml')
    # each_save_path_jpg = os.path.join(save_dir, each_save_name)
    #
    if os.path.exists(each_save_path_xml):
        print("* ignore img have res already")
        return
    #

    # todo 下面的逻辑使用一个循环进行维护

    if "jyzZB" in model_list:
        jyzZB_dete_res = dete_jyzZB(model_dict, data)
        if jyzZB_dete_res:
            dete_res_all += jyzZB_dete_res

    if "nc" in model_list:
        nc_dete_res = dete_nc(model_dict, data)
        if nc_dete_res:
            dete_res_all += nc_dete_res

    if "fzc" in model_list:
        fzc_dete_res = dete_fzc(model_dict, data)
        if fzc_dete_res:
            dete_res_all += fzc_dete_res

    if "kkxTC" in model_list or "kkxQuiting" in model_list or "kkxRust" in model_list:
        kkx_dete_res = dete_kkx(model_dict, data)
        if kkx_dete_res:
            dete_res_all += kkx_dete_res

    if "jyhQX" in model_list:
        jyhQX_dete_res = dete_jyhQX(model_dict, data)
        if jyhQX_dete_res:
            dete_res_all += jyhQX_dete_res

    if "xjQX" in model_list:
        xjQX_dete_res = dete_xjQX(model_dict, data)
        if xjQX_dete_res:
            dete_res_all += xjQX_dete_res

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
    if len(dete_res_all) > 0:
        dete_res_all.print_as_fzc_format()
    else:
        print([])


    # dete_res_all.draw_dete_res(each_save_path_jpg)
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
    dete_mode = args.deteMode
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
    all_model_list = ['nc', 'jyzZB', 'fzc', 'fzcRust', 'kkxTC', 'kkxQuiting', 'xjQX', 'jyhQX', 'kkxClearance']
    # all_model_list = ['nc', 'jyzZB', 'fzc', 'fzcRust', 'kkxTC', 'kkxQuiting']
    # all_model_list = ['xjQX', 'jyhQX']

    all_model_dict = all_model_restore(args, scriptName, all_model_list)
    print("* warm model success ")

    start_time = time.time()

    #
    while True:
        img_path_list, each_img_dir = get_img_path_list_from_sign_dir(sign_txt_path, img_dir, dete_mode)
        print("dete model_list : {0}".format(len(img_path_list)))
        # dete
        for each_img_path in img_path_list:
            dete_img_index += 1
            # -------------------------------------
            save_dir = os.path.join(output_dir, "save_res")
            os.makedirs(save_dir, exist_ok=True)
            each_save_name = os.path.split(each_img_path)[1]
            each_save_path_xml = os.path.join(save_dir, each_save_name[:-4] + '.xml')
            #
            if os.path.exists(each_save_path_xml):
                print("* ignore img have res already : {0}".format(each_save_path_xml))
                continue
            else:
                print("* {0} : {1}".format(dete_img_index, each_img_path))

            # -------------------------------------

            try:
                each_model_list = all_model_list
                try:
                    each_dete_res = model_dete(each_img_path, all_model_dict, each_model_list)
                except Exception as e:
                    print(e)
                    print(e.__traceback__.tb_frame.f_globals["__file__"])
                    print(e.__traceback__.tb_lineno)
                    # move error img to assign dir
                    # shutil.move(each_img_path, os.path.join(sign_error_dir, os.path.split(each_img_path)[1]))
                #
                # if os.path.exists(each_img_path):
                #     os.remove(each_img_path)
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
        exit()
        # print("* wait for 2s")
        # time.sleep(2)

        # todo 在 sign 文件夹中增加 两个表示检测完毕的  txt
        end_time = time.time()
        # add file to output_dir
        res_txt = os.path.join(sign_dir, "res_txt")
        os.makedirs(res_txt, exist_ok=True)
        txt_path = os.path.join(res_txt, "{0}.txt".format(script_index))
        with open(txt_path, 'w') as txt_file:
            txt_file.write('done')




