# -*- coding: utf-8  -*-
import json
import configparser
import numpy as np
import os
import torch
import random
from Crypto.Cipher import AES
import struct
import sys
import cv2
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except, timeStamp
from ..detect_utils.cryption import salt, decrypt_file
from ..detect_libs.abstractBase import detection
from torch.backends import cudnn
from torchvision import transforms
from ..mmdet_libs.mmseg.apis import inference_segmentor, init_segmentor

class mmsegDetection(detection):

    def __init__(self, args, objName=None, scriptName=None):
        super(mmsegDetection, self).__init__(objName, scriptName)
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
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuID)

    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName = self.cf.get('common', 'model')
        self.debug = self.cf.getboolean("common", 'debug')
        # fixme 这边要和之前的规范进行统一
        self.tfmodelName = self.cf.get(self.objName, 'modelname')
        self.CLASSES = tuple(self.cf.get(self.objName, 'classes').strip(',').split(','))
        self.VISIBLE_CLASSES = tuple(self.cf.get(self.objName, 'visible_classes').strip(',').split(','))
        self.encryption = self.cf.getboolean("common", 'encryption')
        self.model_type = self.cf.get(self.objName, 'model_type')
        self.compose_config()

    def compose_config(self):
        self.config = {}
        self.config['num_classes'] = len(self.CLASSES) + 1

    @try_except()
    def model_restore(self):
        self.log.info('===== model restore start =====')
        # 加密模型

        if self.encryption:
            model_path = self.dncryptionModel()
        else:
            model_path = os.path.join(self.modelPath, self.tfmodelName)
        self.log.info("self.encryption:", self.encryption)

        # fixme 根据配置文件，进行修改
        self.model = init_segmentor(self.config, self.model_type, model_path,device='cuda:'+str(self.gpuID))

        # 这边是不是要返回参数
        #self.warmUp()
        self.log.info('model restore end')

        # 删除解密的模型
        if self.encryption:
            os.remove(model_path)
            self.log.info('delete dncryption model successfully! ')
        return

    @try_except()
    def warmUp(self):
        #im = 128 * np.ones((1440, 2560, 3), dtype=np.uint8)
        im = 128 * np.random.rand(1440, 2560, 3)
        im = im.astype(np.uint8)
        
        self.detect(im, 'warmup.jpg')

    @timeStamp()
    @try_except()
    def detect(self, im, image_name="default.jpg", resize_ratio=1):
        self.log.info('=========================')
        self.log.info(self.modelName + ' detection start')
        h,w,_ = im.shape   
        resized_img = cv2.resize(im,(w*resize_ratio,h*resize_ratio))
        with torch.no_grad():
            result = inference_segmentor(self.model, resized_img)
        result = torch.tensor(np.array(result))
        result.squeeze_()
        result = result.numpy()
        # 这里输出的是单通道
        masks = self.classifyMask(result)
        return masks

    @try_except()
    def classifyMask(self,mask):
        result = {}
        print(self.CLASSES)
        for i in range(1,len(self.CLASSES)+1):
            label_mask = np.zeros((mask.shape[0],mask.shape[1]))
            pos = np.argwhere(mask==i)
            print(pos)
            label_mask[mask==i] = 1
            print(self.CLASSES[i-1])
            result[self.CLASSES[i-1]] = label_mask 
        return result

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
