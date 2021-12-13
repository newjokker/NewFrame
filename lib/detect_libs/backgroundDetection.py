#! /usr/bin/env python
import os
import cv2,copy
import torch
import configparser
import warnings
import numpy as np
from PIL import Image
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except
from ..detect_libs.abstractBase import detection
from ..detect_utils.cryption import salt, decrypt_file
import torch.nn.functional as F
from ..JoTools.txkjRes.deteRes import DeteRes
from ..JoTools.txkjRes.deteObj import DeteObj

from ..background_libs.unetBase import Unet as unet


class BackgroundDetection(detection):
    def __init__(self, args, objName, scriptName):
        super(BackgroundDetection, self).__init__(objName, scriptName)
        self.readArgs(args)
        self.readCfg()
        self.log = detlog(self.modelName, self.objName, self.logID)

        cuda_select = 'cuda:'+str(self.gpuID)
        self.device = torch.device(cuda_select if torch.cuda.is_available() else "cpu")
        self.blend = True
        self.unet= unet
        # 颜色是BGR格式的，为后续输出方便，这里背景为0，其他所有类别均填充同样的数值
        self.colors = [(0, 0, 0), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1),
                       (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1),
                       (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)]

    def readArgs(self, args):
        self.portNum = args.port
        self.gpuID = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host = args.host
        self.logID = args.logID
        
    @try_except()
    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName   = self.cf.get('common', 'model')
        self.encryption  = self.cf.getboolean("common", 'encryption')
        self.debug       = self.cf.getboolean("common", 'debug')
        self.tfmodelName = self.cf.get(self.objName,'modelname')
        self.normSize    = self.cf.getint(self.objName, 'norm_size')

        try:
            self.safe_rate_threshold   = self.cf.getfloat(self.objName,'safe_rate_threshold')
        except Exception as e:
            pass
        self.num_classes = self.cf.getint(self.objName, 'num_classes')
        self.model_image_size = (self.normSize, self.normSize, 3)
        try:
            self.run_mode = self.cf.get('common', 'run_mode')
        except:
            self.run_mode = 'crop'

    @try_except()
    def model_restore(self):
        self.log.info('===== model restore start =====')


        # 加密模型
        if self.encryption:

            model_path = self.dncryptionModel()
        else:
            model_path = os.path.join(self.modelPath, self.tfmodelName)

        self.net = self.unet(num_classes=self.num_classes, in_channels=self.model_image_size[-1]).eval()
        state_dict = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.model = self.net.to(self.device)

        self.warmUp()
        self.log.info("load complete")
        print("load complete")

        # 删除解密的模型
        if self.encryption:
            os.remove(model_path)
            self.log.info('delete dncryption model successfully! ')

    @try_except()
    def warmUp(self):
        self.model(torch.zeros(1, 3, self.normSize, self.normSize).to(self.device).type_as(next(self.model.parameters())))

    @staticmethod  # 该函数并未实际使用self属性，故而强迫症犯了，在此声明了静态方法。只是为了消除IDE的给出了警告，
    def letterbox_image(image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image, nw, nh

    @try_except()
    def detect_image(self, image):
        image = image.convert('RGB')  # 在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        old_img = copy.deepcopy(image)  # 对输入图像进行一个备份，后面用于抠图
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # 进行不失真的resize，添加灰条，进行图像归一化
        image, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        images = [np.array(image) / 255]
        images = np.transpose(images, (0, 3, 1, 2))

        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor)
            images = images.to(self.device)

            pr = self.net(images)[0]  # 图片传入网络进行预测
            # 取出每一个像素点的种类
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            # 将灰条部分截取掉
            pr = pr[int((self.model_image_size[0] - nh) // 2):int((self.model_image_size[0] - nh) // 2 + nh),
                    int((self.model_image_size[1] - nw) // 2):int((self.model_image_size[1] - nw) // 2 + nw)]

        # 创建一副新图，并根据每个像素点的种类赋予颜色.
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')

        # -------------惊喜吧，这里才是真正的输出。----------------
        img_return = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))  # 格式转来转去的不嫌烦

        if self.blend:  # 将新图片和原图片混合，留下了观察分割结果的变量，但是并不是网络的输出
            image = Image.blend(old_img, img_return, 0.5)

        return img_return



    @try_except()
    def dncryptionModel(self):
        print('*********  get in self.encryption **********')
        if not os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)

        # 解密模型
        name, ext = os.path.splitext(self.tfmodelName)
        model_origin_name = name + ext
        model_locked_name = name + "_locked" + ext
        origin_Fmodel = os.path.join(self.cachePath, model_origin_name)
        locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
        decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        # 解密后的模型
        tfmodel = os.path.join(self.cachePath, self.tfmodelName)
        return tfmodel
