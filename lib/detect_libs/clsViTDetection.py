import os
import cv2
import configparser
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ..viT_libs.vit_model import vit_base_patch16_224_in21k as create_model
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import *
from ..detect_libs.abstractBase import detection
from ..detect_utils.cryption import decrypt_file, salt


class ClsViTDetection(detection):
    def __init__(self, args, objName, scriptName):
        super(ClsViTDetection, self).__init__(objName, scriptName)
        self.readArgs(args)
        self.readCfg()
        self.log = detlog(self.modelName, self.objName, self.logID)

    def readArgs(self, args):
        self.portNum  = args.port
        self.gpuID    = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host     = args.host
        self.logID    = args.logID
        cuda_info = "cuda:"+str(self.gpuID)
        self.device = torch.device(cuda_info)

    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName   = self.cf.get('common', 'model')
        self.encryption  = self.cf.getboolean("common", 'encryption')
        self.debug       = self.cf.getboolean("common", 'debug')

        self.tfmodelName = self.cf.get(self.objName,'modelName')
        self.normSize    = self.cf.getint(self.objName, 'norm_size')
        self.confThresh  = self.cf.getfloat(self.objName,'conf_threshold')
        self.num_classes = self.cf.getint(self.objName, 'num_classes')

    @try_except()
    def model_restore(self):
        torch.cuda.empty_cache()

        if self.encryption:
            tfmodel = self.dncryptionModel()
        else:
            tfmodel = os.path.join(self.modelPath, self.tfmodelName)
        self.log.info(tfmodel)
        print('=====> viT tfmodel = ',tfmodel)

        # 图像预处理部分
        self.data_transform = transforms.Compose(
            [transforms.Resize(self.normSize),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


        self.model = create_model(num_classes=self.num_classes, has_logits=False).to(self.device)
        self.model.load_state_dict(torch.load(tfmodel, map_location=self.device))
        self.model.eval()

        print('=====> viT warmUp starts... = ')
        self.warmUp()
        print('=====> viT warmUp done = ')


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
            image = Image.fromarray(im.astype('uint8')).convert('RGB')  # np.array==>Image
            img = self.data_transform(image)
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                output = torch.squeeze(self.model(img.to(self.device))).cpu()
                proba= max(torch.softmax(output, dim=0).numpy())  # 概率
                label = torch.argmax(torch.softmax(output, dim=0)).numpy()  # 输出为: 0 or 1 or 2
            return label, proba














