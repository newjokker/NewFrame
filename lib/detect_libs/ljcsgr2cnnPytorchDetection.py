# -*- coding: utf-8 -*-
# @Author: chencheng
# @Date:   2021-08-19 11:43:54
# @Last Modified by:   chencheng
# @Last Modified time: 2021-08-19 20:24:00
# @E-mail: chencheng@tuxingkeji.com
# 

import os
import torch
from torchvision import transforms as T
import cv2, copy
import json
import math
import os, sys
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, "..", "r2cnnPytorch_libs")
sys.path.insert(0, lib_path)
import configparser
import numpy as np
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except, timeStamp
from ..detect_utils.cryption import salt, decrypt_file
from ..detect_utils.utils import mat_inter
from ..detect_libs.abstractBase import detection
from ..detect_libs.r2cnnPytorchDetection import R2cnnDetection

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import  DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.config import cfg

from Crypto.Cipher import AES
import struct
from xml.etree.ElementTree import Element, ElementTree, tostring, SubElement
import xml.etree.ElementTree as ET

class LjcSgR2cnnDetection(R2cnnDetection):
    def __init__(self, args, objName, scriptName):
        super(LjcSgR2cnnDetection, self).__init__(args, objName, scriptName)
        self.resizedImgPath = self.getTmpPath('resizedImg')
        self.label_list1 = ['YuJiaoShiXJ', 'XuanChuiXianJia', 'UGuaHuan', 'XinXingHuan','TXXJ','LSXXJ','XieXingXJ', 'BGXJ', 'SXJ', 'Zhongchui', 'other2', 'ULuoShuan'] #单金具截图
        self.label_list2 = ['XieXingXJ', 'XinXingHuan', 'BGXJ'] #塔顶，两两互相组合截图
        #self.label_list3 = ['YuJiaoShiXJ', 'XuanChuiXianJia', 'TXXJ', 'SXJ', 'XieXingXJ'] #与防振锤进行联合截图

    def postProcess(self, im, name, detectBoxes):
        boxes = []
        labels = []
        results = []
        h, w, _ = im.shape
        try:
            for box in detectBoxes:
                xmin, xmax, ymin, ymax = self.minirect2(box[0:4], im.shape[0], im.shape[1])
                label = box[-1]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label) 
          
            for index, (box, label) in enumerate(list(zip(boxes, labels))):
                xmin, ymin, xmax, ymax = box
                resizedName = name + '_' + label + '_' + str(index) + '_' + str(xmin) + '_' + str(ymin) + '_' + str(xmax) + '_' + str(ymax)
                xExtend, yExtend = self.getExtendForLjc(label)
                xmin, ymin, xmax, ymax = self.getResizedImgSize(im, xmin, ymin, xmax, ymax, xExtend, yExtend)
                ww = xmax - xmin
                hh = ymax - ymin

                xmintmp = max(xmin - ww, 1)
                xmaxtmp = min(xmax + ww, w)
                ymintmp = max(ymin - hh, 1)
                ymaxtmp = min(ymax + hh, h)
                resultImg = im[int(ymintmp):int(ymaxtmp), int(xmintmp):int(xmaxtmp)]

                ##-------单金具截图
                if label in self.label_list1:
                    results.append({'resizedName': resizedName, 'label': label, 'index': str(index), 'xmin': xmintmp, 'ymin': ymintmp,'xmax': xmaxtmp, 'ymax': ymaxtmp})
                    cv2.imwrite(os.path.join(self.resizedImgPath, resizedName + '.jpg'),resultImg)

                ##-------塔顶金具截图
                for index_s, (box_s, label_s) in enumerate(list(zip(boxes, labels))):
                    xmin_s, ymin_s, xmax_s, ymax_s = box_s
                    if (label in self.label_list2) and (label_s in self.label_list2) and (index != index_s):
                        
                        xExtend_s, yExtend_s = self.getExtendForLjc(label_s)
                        xmin_s, ymin_s, xmax_s, ymax_s = self.getResizedImgSize(im, xmin_s, ymin_s, xmax_s, ymax_s, xExtend_s, yExtend_s)
                        xmintmp_s = max(xmin_s - ww, 1)
                        xmaxtmp_s = min(xmax_s + ww, w)
                        ymintmp_s = max(ymin_s - hh, 1)
                        ymaxtmp_s = min(ymax_s + hh, h)

                        crop_xmin = min(xmintmp, xmintmp_s)
                        crop_xmax = max(xmaxtmp, xmaxtmp_s)
                        crop_ymin = min(ymintmp, ymintmp_s)
                        crop_ymax = max(ymaxtmp, ymaxtmp_s)

                        resizedName_s = name + '_' + label + '_' + str(index) + '_' + label_s + '_' + str(index_s) + '_' + str(crop_xmin) + '_' + str(crop_ymin) + '_' + str(crop_xmax) + '_' + str(crop_ymax) 
                        resultImg_s = im[int(crop_ymin):int(crop_ymax), int(crop_xmin):int(crop_xmax)]
                        results.append({'resizedName': resizedName_s, 'label': label+'_'+label_s, 'index': str(index)+'_'+str(index_s), 'xmin':crop_xmin, 'ymin': crop_ymin, 'xmax': crop_xmax, 'ymax': crop_ymax})
                        cv2.imwrite(os.path.join(self.resizedImgPath, resizedName_s + '.jpg'), resultImg_s)
        except Exception as e:
            self.log.info(e)
            self.log.info(e.__traceback__.tb_frame.f_globals["__file__"])
            self.log.info(e.__traceback__.tb_lineno)
            print(e)
        
        return results
    

    @try_except()
    def getExtendForLjc(self, label):
        switchTable = {
            "Zhongchui": (150, 150),
            "xinxingHuan": (150, 200),
            "BGXJ": (150, 200),
            "XieXingXJ": (150, 200),
            "YuJiaoShiXJ": (200, 200),

        }
        if label in switchTable.keys():
            return switchTable[label]
        else:
            return (150, 150)


    @try_except()
    def getResizedImgSize(self, im, xmin, ymin, xmax, ymax, xExtend, yExtend):
        width = im.shape[1]
        height = im.shape[0]
        xmin_r = max(xmin - xExtend, 0)
        xmax_r = min(xmax + xExtend, width)
        ymin_r = max(ymin - yExtend, 0)
        ymax_r = min(ymax + yExtend, height)
        return xmin_r, ymin_r, xmax_r, ymax_r
