import os
import cv2
import configparser
import numpy as np
import torch
from .saturn_vgg import vgg19
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import *
from ..detect_libs.abstractBase import detection
from ..detect_utils.cryption import decrypt_file, salt


class ClsDetectionPyTorch(detection):
    def __init__(self, args, objName, scriptName):
        super(ClsDetectionPyTorch, self).__init__(objName, scriptName)
        self.readArgs(args)
        self.readCfg()
        self.log = detlog(self.modelName, self.objName, self.logID)

    def readArgs(self, args):
        self.portNum  = args.port
        self.gpuID    = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host     = args.host
        self.logID    = args.logID
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuID)

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
        torch.cuda.empty_cache()

        if self.encryption:
            tfmodel = self.dncryptionModel()
        else:
            tfmodel = os.path.join(self.modelPath, self.tfmodelName)
        self.log.info(tfmodel)

        self.state_dict = torch.load(tfmodel)
        n_classes = self.state_dict["classifier.3.bias"].shape[0]
        self.model = vgg19(pretrained=False, num_classes=n_classes, norm_size=self.normSize)
        self.model.load_state_dict(self.state_dict)
        self.model.cuda().eval()
        del self.state_dict

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

    @timeStamp()
    @try_except()
    def detect(self, im, image_name="default.jpg"):
        if im is None:
            self.log.info("【Waring】:"+ image_name +"  == None !!!")
            return None, 0

        else:
            image = cv2.resize(im, (self.normSize, self.normSize))
            image = image.astype("float") / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().cuda()

            with torch.no_grad():
                out = self.model(image)
                out = torch.softmax(out, dim=1)
            out.cpu()
            torch.cuda.empty_cache()

            proba, label = out.max(dim=1)
            label, proba = str(label.item()), proba.item()
            return label, proba














