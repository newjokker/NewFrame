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
import copy
#
from lib.detect_libs.yolov5Detection import YOLOV5Detection
from lib.detect_utils.timer import Timer
from lib.detect_libs.fasterDetectionPyTorch import FasterDetectionPytorch
from lib.detect_libs.vggClassify import VggClassify
from lib.detect_libs.clsDetectionPyTorch import ClsDetectionPyTorch
from lib.detect_libs.ljcY5Detection import LjcDetection
from lib.detect_libs.kkgY5Detection import KkgDetection
from lib.detect_libs.clsViTDetection import ClsViTDetection                             # kkxTC vit
#
from lib.JoTools.txkjRes.resTools import ResTools
from lib.JoTools.utils.FileOperationUtil import FileOperationUtil
from lib.JoTools.utils.CsvUtil import CsvUtil
from lib.JoTools.txkjRes.deteRes import DeteRes
from lib.JoTools.txkjRes.deteObj import DeteObj
from lib.JoTools.txkjRes.deteAngleObj import DeteAngleObj
from lib.JoTools.utils.JsonUtil import JsonUtil

#
# load dict
from fangtian_info_dict import M_dict, M_model_list, key_M_dict, tag_code_dict
# time analysis
from JoTools.utils.DecoratorUtil import DecoratorUtil



@DecoratorUtil.time_this
def all_model_restore(args, scriptName, model_list):
    """模型预热"""
    model_dict = {}

    if "jyzZB" in model_list:
        model_jyzZB_1 = YOLOV5Detection(args, "jyz", scriptName)
        model_jyzZB_1.model_restore()
        #
        model_jyzZB_2 = YOLOV5Detection(args, "jyzzb", scriptName)
        model_jyzZB_2.model_restore()
        #
        model_dict["model_jyzZB_1"] = model_jyzZB_1
        model_dict["model_jyzZB_2"] = model_jyzZB_2

    if "nc" in model_list:
        model_nc = YOLOV5Detection(args, "nc", scriptName)
        model_nc.model_restore()
        #
        model_dict["model_nc"] = model_nc

    if "fzcRust" in model_list or "fzc" in model_list:
        model_fzc_1 = FasterDetectionPytorch(args, "fzc_step_one", scriptName)
        model_fzc_1.model_restore()
        #
        model_fzc_2 = VggClassify(args, "fzc_step_new", scriptName)
        model_fzc_2.model_restore()
        #
        model_fzc_rust = ClsViTDetection(args, "fzc_rust", scriptName)
        model_fzc_rust.model_restore()
        #
        model_dict["model_fzc_1"] = model_fzc_1
        model_dict["model_fzc_2"] = model_fzc_2
        model_dict["model_fzc_rust"] = model_fzc_rust

    if "fncDK" in model_list:
        model_fnc = YOLOV5Detection(args, "fnc", scriptName)
        model_fnc.model_restore()
        #
        model_dict["model_fnc"] = model_fnc

    if "kkxTC" in model_list or "kkxQuiting" in model_list or "kkxClearence" in model_list:
        # kkx
        model_kkxTC_1 = LjcDetection(args, "kkxTC_ljc", scriptName)
        model_kkxTC_1.model_restore()
        #
        model_kkxTC_2 = KkgDetection(args, "kkxTC_kkx", scriptName)
        model_kkxTC_2.model_restore()
        #
        model_kkxTC_3 = ClsViTDetection(args, "kkxTC_lm_cls_vit", scriptName)
        model_kkxTC_3.model_restore()
        # kkxQuiting
        model_kkxQuiting = ClsViTDetection(args, "kkxQuiting_cls", scriptName)
        model_kkxQuiting.model_restore()
        # kkxClearence
        model_kkxClearence = ClsViTDetection(args, "kkxClearence", scriptName)
        model_kkxClearence.model_restore()
        #
        model_dict["model_kkxTC_1"] = model_kkxTC_1
        model_dict["model_kkxTC_2"] = model_kkxTC_2
        model_dict["model_kkxTC_3"] = model_kkxTC_3
        model_dict["model_kkxQuiting"] = model_kkxQuiting
        model_dict["model_kkxClearence"] = model_kkxClearence

    if "waipo" in model_list:
        model_waipo = YOLOV5Detection(args, "waipo", scriptName)
        model_waipo.model_restore()
        #
        model_dict["model_waipo"] = model_waipo

    if "ljcRust" in model_list:
        model_ljc_rust_1 = YOLOV5Detection(args, "ljc_rust_one", scriptName)
        model_ljc_rust_1.model_restore()
        #
        model_ljc_rust_2 = ClsDetectionPyTorch(args, "ljc_rust_two", scriptName)
        model_ljc_rust_2.model_restore()
        #
        model_dict["model_ljc_rust_1"] = model_ljc_rust_1
        model_dict["model_ljc_rust_2"] = model_ljc_rust_2

    if "jyhQX" in model_list:
        # jyhQX
        from lib.detect_libs.r2cnnPytorchDetection import R2cnnDetection
        from lib.detect_libs.jyhDeeplabDetection import jyhDeeplabDetection
        #
        model_jyhQX_1 = YOLOV5Detection(args, "jyhqx_one", scriptName)
        model_jyhQX_1.model_restore()
        #
        model_jyhQX_2 = R2cnnDetection(args, "jyhqx_two", scriptName)
        model_jyhQX_2.model_restore()
        #
        model_jyhQX_3 = jyhDeeplabDetection(args, "jyhqx_three", scriptName)
        model_jyhQX_3.model_restore()
        #
        model_dict["model_jyhqx_1"] = model_jyhQX_1
        model_dict["model_jyhqx_2"] = model_jyhQX_2
        model_dict["model_jyhqx_3"] = model_jyhQX_3

    if "xjQX" in model_list:
        # xjQX
        from lib.detect_libs.xjdectR2cnnPytorchDetection import XjdectR2cnnDetection
        from lib.detect_libs.xjDeeplabDetection import xjDeeplabDetection
        #
        model_xjQX_1 = XjdectR2cnnDetection(args, "xjQX_ljc", scriptName)
        model_xjQX_1.model_restore()
        #
        #model_xjQX_2 = xjDeeplabDetection(args, "xjQX_deeplab", scriptName)
        #model_xjQX_2.model_restore()
        #
        model_dict["model_xjQX_1"] = model_xjQX_1
        #model_dict["model_xjQX_2"] = model_xjQX_2

    if "xjDP" in model_list or "ljcRust" in model_list:
        from lib.detect_libs.r2cnnPytorchDetection import R2cnnDetection
        from lib.detect_libs.rustBackgroundDetection import RustBackgroundDetection
        #
        model_xjDP_ljc = R2cnnDetection(args, "xjDP_ljc", scriptName)
        model_ljcRust_rust = RustBackgroundDetection(args, 'ljcRust_rust', scriptName)
        #
        model_xjDP_ljc.model_restore()
        model_ljcRust_rust.model_restore()
        #
        model_dict['model_xjDP_ljc'] = model_xjDP_ljc
        model_dict['model_ljcRust_rust'] = model_ljcRust_rust


    return model_dict





