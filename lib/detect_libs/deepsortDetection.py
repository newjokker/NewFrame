# -*- coding: utf-8 -*-
# @Author: chencheng
# @Date:   2021-05-29 14:27:23
# @Last Modified by:   chencheng
# @Last Modified time: 2021-06-09 15:20:23
# @E-mail: chencheng@tuxingkeji.com

import os
import cv2
import json
import torch
import configparser
import numpy as np

from ..detect_libs.abstractBase import detection
from ..detect_utils.log import detlog
from ..detect_utils.utils import isBoxExist
from ..detect_utils.cryption import salt, decrypt_file
from ..detect_utils.tryexcept import try_except,timeStamp

from ..deep_sort_libs.utils.parser import get_config
from ..deep_sort_libs.deep_sort.deep.feature_extractor import Extractor
from ..deep_sort_libs.deep_sort.sort.tracker import Tracker
from ..deep_sort_libs.deep_sort.sort.detection import Detection as ds_detect
from ..deep_sort_libs.deep_sort.sort.preprocessing import non_max_suppression
from ..deep_sort_libs.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric


class DeepSortDetection(detection):
    def __init__(self, args, objName, scriptName):
        super(DeepSortDetection, self).__init__(objName, scriptName)

        self.readArgs(args)
        self.readCfg()
        self.device = torch.device("cuda:" + str(self.gpuID)) if self.gpuID else 'cpu'
        self.log = detlog(self.modelName, self.objName, self.logID)

        self.max_cosine_distance = self.max_dist
        self.metric = NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(
            self.metric, max_iou_distance = self.max_iou_distance, max_age = self.max_age, n_init = self.n_init)

    def readArgs(self, args):
        self.portNum  = args.port
        self.gpuID    = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host     = args.host
        self.logID    = args.logID

    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName        = self.cf.get('common', 'model')
        self.encryption       = self.cf.getboolean("common", 'encryption')
        self.debug            = self.cf.getboolean("common", 'debug')

        self.model_path       = self.cf.get(self.objName, 'modelName')
        self.max_dist         = eval(self.cf.get(self.objName, 'max_dist'))
        self.min_confidence   = eval(self.cf.get(self.objName, 'min_confidence'))
        self.nms_max_overlap  = eval(self.cf.get(self.objName, 'nms_max_overlap'))
        self.max_iou_distance = eval(self.cf.get(self.objName, 'max_iou_distance'))
        self.max_age          = eval(self.cf.get(self.objName, 'max_age'))
        self.n_init           = eval(self.cf.get(self.objName, 'n_init'))
        self.nn_budget        = eval(self.cf.get(self.objName, 'nn_budget'))

    @try_except()
    def model_restore(self):
        self.log.info('===== model restore start =====')
        # 加密模型
        if self.encryption:
            model_path = self.dncryptionModel()
        else:
            model_path = os.path.join(self.modelPath, self.model_path)
        self.log.info('model_path is:', model_path)

        self.extractor = Extractor(model_path, self.device)

        if self.encryption:
            self.delDncryptionModel() 
        
        self.warmUp()
        self.log.info('===== model restore end =====')
        return

    @try_except()
    def dncryptionModel(self):
        if False == os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)
        ### 解密模型 ####
        name, ext = os.path.splitext(self.model_path)
        model_origin_name = name + ext
        model_locked_name = name + "_locked" + ext
        origin_Fmodel = os.path.join(self.cachePath, model_origin_name)
        locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
        decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        ### 解密后的模型 ####
        tfmodel = os.path.join(self.cachePath, self.model_path)
        return tfmodel

    @try_except()
    def delDncryptionModel(self):
        # 删除解密的模型
        # os.remove(self.model_path)
        # self.log.info('delete dncryption model successfully! ')
        return

    @try_except()
    def warmUp(self):
        im = 128 * np.ones((4, 512, 512, 3), dtype=np.uint8)
        feature = self.extractor(im)
        print(f'deepsort warmUp output: {feature}')
        return

    def update(self, bbox_xywh, confidences, ori_img):

        # fixme 这个的作用是什么？

        # 卡尔曼滤波

        self.height, self.width = ori_img.shape[:2]

        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [ds_detect(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence]
        print(f'deepsort detection: {detections}')

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
