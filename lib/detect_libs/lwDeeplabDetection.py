# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import cv2
import numpy as np
from skimage import morphology
from ..detect_utils.tryexcept import try_except
from ..detect_libs.deepLabSegmentPyTorch import DeepLabSegmentPytorch

class lwDeeplabDetection(DeepLabSegmentPytorch):
    def __init__(self, args, objName=None, scriptName=None):
        super(lwDeeplabDetection, self).__init__(args,objName, scriptName)

    @try_except()
    def detectSOUT(self, im, image_name="default.jpg", resize_ratio=1):
        raw = self.detect(im,image_name,resize_ratio)
        raw_lw_points = raw['block']
        #print(raw_lw_points.shape)
        post_lw_points = self.filterPoints(raw_lw_points)
        return post_lw_points

    @try_except()
    def filterPoints(self,raw_points):
        self.log.info('filterPoint start')
        k1 = np.ones((5, 5), np.uint8)  # (2560*1440 --> 5*5) (1280*720 --> 3*3)
        k2 = np.ones((30, 30), np.uint8)  # 连接 2*X 像素外的点, 推荐 20--60
        self.log.info('rgb.shape:',raw_points.shape)
        rgb = raw_points
        rgb = cv2.erode(rgb, k1)  # 腐蚀操作去杂点毛边
        self.log.info('erode')
        rgb = cv2.dilate(rgb, k2)  # 膨胀操作使直线上离散的线段连接
        self.log.info('dilate')
        rgb[rgb == 255] = 1  # 二值化处理
        skeleton0 = morphology.skeletonize(rgb)  # 骨架提取
        self.log.info('skeletion')
        result = np.argwhere(skeleton0 != 0)
        self.log.info('result.shape:',result.shape)
        length = len(result)
        index = np.linspace(0,length,endpoint = False,num=int(length/10),dtype=int)
        self.log.info('filterPoint end')
        points = result[index]
        self.log.info(points.shape)
        points = np.array(points[:,0:2])
        self.log.info(type(points))
        self.log.info(points.shape)
        final = points[:,[1,0]]
        return final
