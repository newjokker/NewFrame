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


# todo 增加 resize 功能，让用于分割的图像都是差不多大小


class MaskSegmentPytorch(detection):

    def __init__(self, args, objName=None, scriptName=None):
        super(MaskSegmentPytorch, self).__init__(objName, scriptName)
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
        self.detectSOUT(image=im, image_name='warmup.jpg')
        print("* 模型加载完毕")

    @torch.no_grad()
    @try_except()
    def detectSOUT(self, path=None, image=None, image_name="default.jpg", return_each_obj=False, max_length=1000.0):
        """规范返回值的格式，可以是所有元素都在一个层上，可以是每一个元素在一个层上, 要是 return_each_obj 是 True 返回多层"""

        try:

            if path is None and image is None:
                raise ValueError("path and image cannot be both None")

            dete_res = DeteRes()
            dete_res.img_path = path
            dete_res.file_name = image_name

            if image is None:
                image = dete_res.get_img_array()

            # resize to assign size
            width, height = image.shape[:2]

            ratio = max([width, height]) / max_length
            image = cv2.resize(image, (int(height/ratio), int(width/ratio)))

            img_tensor = torch.from_numpy(image / 255.).permute(2, 0, 1).float().cuda()
            out = self.net([img_tensor])

            # get mask
            mask = out[0]["masks"].cpu().detach().numpy()
            mask = np.squeeze(mask)
            mask = mask > self.confThresh
            mask = mask.astype(np.uint8)

            # init dete res
            dete_res = DeteRes()

            # exit when no obj
            if mask.ndim != 3 or mask.shape[0] == 0:
                return dete_res, np.zeros((width, height))


            # if distinguish each object
            if return_each_obj:
                value_index = 1
                for i in range(mask.shape[0]):
                    mask[value_index - 1, :, :] *= value_index
                    value_index += 1

            # get mask
            mask = np.sum(mask, axis=0)
                        
            # resize # todo 这部分要重新写
            mask = np.dstack((mask, mask, mask))
            
            # ===============================  看看别人这块是怎么写的 ========================

            mask = mask.astype(np.uint8)
            mask = cv2.resize(mask, (height, width))
            mask = mask[:,:,0]
                        
            # get box info
            boxes, labels, scores = out[0]['boxes'], out[0]['labels'], out[0]['scores']
            for index, each_box in enumerate(boxes):
                if float(scores[index]) > float(self.confThresh):
                    x1, y1, x2, y2 = int(each_box[0]), int(each_box[1]), int(each_box[2]), int(each_box[3])
                    conf, tag_index = float(scores[index]), str(labels[index].item())
                    # dete_res.add_obj(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, tag='jyz')
                    dete_res.add_obj(x1=x1*ratio, y1=y1*ratio, x2=x2*ratio, y2=y2*ratio, conf=conf, tag='jyz')

            torch.cuda.empty_cache()
            return dete_res, mask
            
        except Exception as e:
            print('GOT ERROR---->')
            print(e)
            print(e.__traceback__.tb_frame.f_globals["__file__"])
            print(e.__traceback__.tb_lineno)

                
        

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
