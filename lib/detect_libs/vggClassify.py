# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import cv2
import torch
import configparser
import numpy as np
import uuid
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import *
from ..detect_libs.abstractBase import detection
from ..detect_utils.cryption import decrypt_file, salt


class VggClassify(detection):

    def __init__(self, args, objName, scriptName):
        super(VggClassify, self).__init__(objName, scriptName)
        self.objName = objName
        self.readArgs(args)
        self.readCfg()
        self.log = detlog(self.modelName, self.objName, self.logID)

    def readArgs(self, args):
        self.portNum = args.port
        self.gpuID = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host = args.host
        self.logID = args.logID
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuID)

    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName = self.cf.get('common', 'model')
        self.encryption = self.cf.getboolean("common", 'encryption')
        self.debug = self.cf.getboolean("common", 'debug')
        self.tfmodelName = self.cf.get(self.objName, 'modelName')
        # self.normSize    = self.cf.getint(self.objName, 'norm_size')
        self.CLASSES = tuple(self.cf.get(self.objName, 'classes').strip(',').split(','))

    @try_except()
    def model_restore(self):
        """加载模型"""
        # model_path = os.path.join(self.modelPath, self.tfmodelName)

        if self.encryption:
            tfmodel = self.dncryptionModel()
        else:
            tfmodel = os.path.join(self.modelPath, self.tfmodelName)

        try:
            self.log.info(tfmodel)
            device = torch.device('cuda')
            self.model = torch.load(tfmodel)
            self.model.to(device)
            self.model.eval()

            self.warmUp()
            if self.encryption:
                os.remove(tfmodel)
                self.log.info('delete dncryption model successfully! ')
        except Exception as e:
            print(e)

        print("模型加载成功")

    @try_except()
    def dncryptionModel(self):
        if False == os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)

        ### 解密模型 ####
        name, ext = os.path.splitext(self.tfmodelName)
        model_locked_name = name + "_locked" + ext
        uuID = uuid.uuid1()
        origin_Fmodel = os.path.join(self.cachePath, str(uuID) + self.tfmodelName)
        locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
        decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        return origin_Fmodel

    @try_except()
    def warmUp(self):
        im = 123 * np.ones((224, 224, 3), dtype=np.uint8)
        self.detect(im, 'warmup.jpg')

    @timeStamp()
    @try_except()
    @torch.no_grad()
    def detect(self, im, image_name="default.jpg"):
        if im is None:
            self.log.info("【Waring】:" + image_name + "  == None !!!")
            return None, 0
        else:
            src_img = cv2.resize(im, (224, 224))
            img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
            img_tensor = torch.unsqueeze(img_tensor, 0)
            out = self.model(img_tensor)

            if hasattr(out, "data"):
                # softmax
                out = torch.nn.functional.softmax(out, 1)
                proba, pred = out.data.max(1, keepdim=True)
                pre = pred.data.item()
                proba = proba.data.item()

                # 清空缓存
                torch.cuda.empty_cache()

                return pre, proba
            else:
                return None, 0


    @timeStamp()
    @try_except()
    @torch.no_grad()
    def detect_new(self, im, image_name="default.jpg"):
        if im is None:
            self.log.info("【Waring】:" + image_name + "  == None !!!")
            return None, 0
        else:
            src_img = cv2.resize(im, (224, 224))
            img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
            img_tensor = torch.unsqueeze(img_tensor, 0)
            out = self.model(img_tensor)

            if hasattr(out, "data"):
                # softmax
                out = torch.nn.functional.softmax(out, 1)
                proba, pred = out.data.max(1, keepdim=True)
                pre = pred.data.item()
                proba = proba.data.item()

                # 清空缓存
                torch.cuda.empty_cache()

                return self.CLASSES[int(pre)], proba
            else:
                return None, 0












