# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import json
import configparser
import numpy as np
import os
import torch
import random
from Crypto.Cipher import AES
import struct
import cv2
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except, timeStamp
from ..detect_utils.cryption import salt, decrypt_file
from ..detect_libs.abstractBase import detection
from torch.backends import cudnn
from ..JoTools.txkjRes.deteRes import DeteRes
from ..JoTools.txkjRes.deteObj import DeteObj


class FasterDetectionPytorch(detection):

    def __init__(self, args, objName=None, scriptName=None):
        super(FasterDetectionPytorch, self).__init__(objName, scriptName)
        self.suffixs = [".pth"]
        self.encryption = False
        self.readArgs(args)
        self.readCfg()
        self.args = args
        self.log = detlog(self.modelName, self.objName, self.logID)

    def readArgs(self, args):
        self.portNum = args.port
        self.gpuID = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host = args.host
        self.logID = args.logID
        # 指定 GPU 的使用
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuID)

    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName = self.cf.get('common', 'model')
        self.debug = self.cf.getboolean("common", 'debug')
        # fixme 这边要和之前的规范进行统一
        self.tfmodelName = self.cf.get(self.objName, 'modelname')
        self.dataset = self.cf.get(self.objName, 'dataset')
        # self.anchorScales = eval(self.cf.get(self.objName, "anchorscales"))
        # self.anchorRatios = eval(self.cf.get(self.objName, "anchorsatios"))
        self.CLASSES = tuple(self.cf.get(self.objName, 'classes').strip(',').split(','))
        self.VISIBLE_CLASSES = tuple(self.cf.get(self.objName, 'visible_classes').strip(',').split(','))
        self.confThresh = self.cf.getfloat(self.objName, 'conf_threshold')
        self.iouThresh = self.cf.getfloat(self.objName, 'iou_threshold')
        # self.compoundCoef = int(self.cf.getfloat(self.objName, 'compound_coef'))
        # self.inputSizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536][self.compoundCoef]
        self.encryption = self.cf.getboolean("common", 'encryption')
        print(self.cf.getboolean("common", 'encryption'))

    @try_except()
    def model_restore(self):
        self.log.info('===== model restore start =====')
        # 加密模型

        if self.encryption:
            model_path = self.dncryptionModel()
        else:
            model_path = os.path.join(self.modelPath, self.tfmodelName)
        print("self.encryption:", self.encryption)
        print(model_path)

        # fixme 下面两个不注释的话，会占用大量的 gpu
        # cudnn.fastest = True
        # cudnn.benchmark = True
        self.net = torch.load(model_path)
        self.net.eval()
        self.net.cuda()
        self.warmUp()
        self.log.info('model restore end')

        # 删除解密的模型
        if self.encryption:
            os.remove(model_path)
            self.log.info('delete dncryption model successfully! ')
        return

    @try_except()
    def warmUp(self):
        im = 128 * np.ones((1000, 1000, 3), dtype=np.uint8)
        self.detect(im, 'warmup.jpg')

    def display(self, preds, imgs):
        """数据展示"""
        res = []
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue
            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                obj = self.CLASSES[preds[i]['class_ids'][j]]
                score = preds[i]['scores'][j]
                res.append([obj, j, int(x1), int(y1), int(x2), int(y2), str(score)])
        return res


    @timeStamp()
    @torch.no_grad()
    @try_except()
    def detect(self, im, image_name="default.jpg", resize_ratio=1):
        self.log.info('=========================')
        self.log.info(self.modelName + ' detection start')
        # im 进行 resize，
        im_height, im_width = im.shape[:2]
        im = cv2.resize(im, (int(im_width*resize_ratio), int(im_height*resize_ratio)))
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(im / 255.).permute(2, 0, 1).float().cuda()
        out = self.net([img_tensor])
        res = []
        # 结果处理并输出
        boxes, labels, scores = out[0]['boxes'], out[0]['labels'], out[0]['scores']

        # 清空缓存
        torch.cuda.empty_cache()

        index = 0
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            score = scores[i].item()
            if score > self.confThresh:
                index += 1
                obj = self.CLASSES[labels[i].item()-1]
                if obj in self.VISIBLE_CLASSES:
                    # 将 resize 后的数据映射回来
                    res.append([obj, index, int(x1/resize_ratio), int(y1/resize_ratio), int(x2/resize_ratio), int(y2/resize_ratio), str(score)])
        return res

    @try_except()
    def detectSOUT(self, path=None, image=None, image_name="default.jpg", output_type='txkj'):
        if path == None and image is None:
            raise ValueError("path and image cannot be both None")
        dete_res = DeteRes()
        dete_res.img_path = path
        dete_res.file_name = image_name
        if image is None:
            image = dete_res.get_img_array()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.detect(bgr, image_name)

        # get deteRes
        assign_id = 0
        for each_obj in results:
            # x1, y1, x2, y2, tag, conf
            dete_res.add_obj(each_obj[2], each_obj[3], each_obj[4], each_obj[5], tag=each_obj[0],
                             conf=float(each_obj[6]), assign_id=assign_id)
            assign_id += 1

        if output_type == 'txkj':
            return dete_res
        elif output_type == 'json':
            pass
        return dete_res

    @try_except()
    def dncryptionModel(self):
        if False == os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)
        ### 解密模型 ####
        name, ext = os.path.splitext(self.tfmodelName)
        model_origin_name = name + ext
        model_locked_name = name + "_locked" + ext
        origin_Fmodel = os.path.join(self.cachePath, model_origin_name)
        locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
        decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        ### 解密后的模型 ####
        tfmodel = os.path.join(self.cachePath, self.tfmodelName)
        return tfmodel



def delete_scrfile(srcfile_path):
    os.remove(srcfile_path)
