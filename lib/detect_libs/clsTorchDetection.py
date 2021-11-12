import os
import cv2
from PIL import Image
import configparser
import numpy as np
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import *
from ..detect_libs.abstractBase import detection
from ..detect_utils.cryption import decrypt_file, salt


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

class ClsDetection(detection):
    def __init__(self, args, objName, scriptName):
        super(ClsDetection, self).__init__(objName,scriptName)
        self.objName = objName
        self.readArgs(args)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuID)
        self.readCfg()
        self.log = detlog(self.modelName, self.objName, self.logID)
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def readArgs(self, args):
        self.portNum  = args.port
        self.gpuID    = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host     = args.host
        self.logID    = args.logID

    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName   = self.cf.get('common', 'model')
        self.encryption  = self.cf.getboolean("common", 'encryption')
        self.debug       = self.cf.getboolean("common", 'debug')
        self.tfmodelName = self.cf.get(self.objName,'modelName')
        self.normSize    = self.cf.getint(self.objName, 'norm_size')
        self.confThresh  = self.cf.getfloat(self.objName,'conf_threshold')

    @try_except()
    def model_restore(self):
        if self.encryption:
            tfmodel = self.dncryptionModel()
        else:
            tfmodel = os.path.join(self.modelPath, self.tfmodelName)
        
        self.log.info(tfmodel)
        
        self.model = torch.load(tfmodel)
        
        self.model.to(self.device)
        self.model.eval()
        self.warmUp()
        if self.encryption:
            os.remove(tfmodel)
            self.log.info('delete dncryption model successfully! ')

    @try_except()
    def dncryptionModel(self):
        if False == os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)

        ### 解密模型 ####
        name, ext = os.path.splitext(self.tfmodelName)
        model_locked_name = name + "_locked" + ext
        origin_Fmodel = os.path.join(self.cachePath, self.tfmodelName)
        locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
        decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        return origin_Fmodel

    @try_except()
    def warmUp(self):
        im = 128 * np.ones((self.normSize, self.normSize, 3),dtype=np.uint8)
        self.detect(im, 'warmup.jpg')
        print('Cls warmUp done')

    @timeStamp()
    @try_except()
    def detect(self, im, image_name="default.jpg"):
        if im is None:
            self.log.info("【Waring】:"+ image_name +"  == None !!!")
            return None, 0
        else:
            
            img_tensor=self.process(im)
            with torch.no_grad():
                out = self.model(img_tensor)
                if len(out)>1:
                    out=out[0]

                if hasattr(out, "data"):
                    #print('have attr data')
                    '''
                    pred = out.data.max(1, keepdim=True)[1]
                    pre = pred.data.item()
                    res = out.data.max(1, keepdim=True)
                    pred, proba = res[1], res[0]
                    pre = pred.data.item()
                    proba = proba.data.item()
                    return pre, proba
                    '''
                    res=torch.softmax(out.data,1)
                    score,y=torch.max(res,1)
                    #print('-+'*10)
                    y=y.item();score=score.item()
                    #print(y,score)
                    if y !=0 and score < self.confThresh :
                        y=0;score=1-score
                    return y,score
                else:
                    #print('no attr data')
                    return None,0

    def process(self,im):
        if self.normSize != -1:
            img = cv2.resize(im, (self.normSize, self.normSize))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(self.device)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        return img_tensor
