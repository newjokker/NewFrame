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
from torchvision import transforms
from ..deeplib_segment_libs.deeplab import DeepLab
from PIL import Image
from torchvision.utils import make_grid
from ..deeplib_segment_libs.dataloaders import custom_transforms as tr
from ..deeplib_segment_libs.dataloaders.utils import decode_seg_map_sequence


class DeepLabSegmentPytorch(detection):

    def __init__(self, args, objName=None, scriptName=None):
        super(DeepLabSegmentPytorch, self).__init__(objName, scriptName)
        #self.suffixs = [".pth"]
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
        self.CLASSES = tuple(self.cf.get(self.objName, 'classes').strip(',').split(','))
        self.VISIBLE_CLASSES = tuple(self.cf.get(self.objName, 'visible_classes').strip(',').split(','))
        self.confThresh = self.cf.getfloat(self.objName, 'conf_threshold')
        #self.iouThresh = self.cf.getfloat(self.objName, 'iou_threshold')
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

        # fixme 根据配置文件，进行修改
        self.net = DeepLab(num_classes=len(self.CLASSES)+1,
                        backbone="mobilenet",
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)

        ckpt = torch.load(model_path, map_location='cpu')
        self.net.load_state_dict(ckpt['state_dict'])
        # 这边是不是要返回参数
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
        #im = 128 * np.ones((1440, 2560, 3), dtype=np.uint8)
        im = 128 * np.random.rand(1440, 2560, 3)
        im = im.astype(np.uint8)
        
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
    @try_except()
    def detect(self, im, image_name="default.jpg", resize_ratio=1):
        self.log.info('=========================')
        self.log.info(self.modelName + ' detection start')

        composed_transforms = transforms.Compose([tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),tr.ToTensor()])
        img = im.copy()
        im = Image.fromarray(im)
        sample = {'image': im, 'label':im.convert('L')}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)
        
        tensor_in = tensor_in.cuda()
                
        self.net.eval()
        
        with torch.no_grad():
            output = self.net(tensor_in)
        rgb_masks = decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),len(self.CLASSES)+1)
        result = self.classifyMask(rgb_masks)
        return result

    @try_except()
    def classifyMask(self,masks):
        result = {}
        for i,mask in enumerate(masks):
            #mask_np = np.array(mask)
            #points = mask_np[:,0:2]
            result[self.CLASSES[i]] = mask#points[:,[1,0]]
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
