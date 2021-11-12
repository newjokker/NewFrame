import os
import cv2
from PIL import Image
import threading
import numpy as np
import warnings
import random
from ..detect_utils.tryexcept import try_except
from .backgroundDetection import BackgroundDetection


class RustBackgroundDetection(BackgroundDetection):
    def __init__(self, args, objName, scriptName):
        super(RustBackgroundDetection, self).__init__(args, objName, scriptName)
        self.resizedImgPath = self.getTmpPath('resizedImg')

    @try_except()
    def cal_color_single(self,img):
        color_angle = []
        img = img / img.max()  # 首先对图像进行归一化
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pix_BGR = img[i, j, :]
                if not np.any(pix_BGR):
                    pass
                else:
                    # 颜色角度：方向角d，高度角h
                    if pix_BGR[0] == 0:
                        d = np.pi / 2
                    else:
                        tan_d = pix_BGR[1] / pix_BGR[0]
                        d = np.arctan(tan_d)
                    if pix_BGR[0] == 0 and pix_BGR[1] == 0:
                        h = np.pi / 2
                    else:
                        tan_h = pix_BGR[2] / np.sqrt(pix_BGR[0] ** 2 + pix_BGR[1] ** 2)
                        h = np.arctan(tan_h)

                    # 为防止奇异值，这里使用了一个诡异的特性:nan ！= nan。应该不会触发。若不慎触发后会给出警告。
                    if h == h and d == d:
                        color_angle.append([d, h])
                    else:
                        warnings.warn('角度出现奇异值')
        return color_angle

    @try_except()
    def cal_rust(self,rate, safe_X, safe_Y, safe_rate, angle):
        isRust_list = []
        for i in range(len(angle)):
            thres = random.random()
            if thres < rate:  # 像素点过多，看不出规律。需进行采样。
                h_hudu = angle[i, 0]  ##表示红色，X轴
                d_hudu = angle[i, 1]  ## Y轴
                h_jiaodu = h_hudu / np.pi * 180
                d_jiaodu = d_hudu / np.pi * 180
                ## 安全区域（非锈蚀）内
                isRust_value = 1 if (h_jiaodu < safe_X) and (d_jiaodu < safe_Y) else 0
                isRust_list.append(isRust_value)
        if len(isRust_list) > 0:
            noRust_rate = sum(isRust_list) / len(isRust_list)
            predict_name = 'clean' if noRust_rate > safe_rate else 'rust'
        else:
            predict_name = 'clean'
        return predict_name
