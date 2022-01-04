# -*- coding: utf-8 -*-
# @Author: zpp
# @Date:   2021-08-13 14:27:23
# @Last Modified by:   zpp
# @Last Modified time: 2021-08-013 14:27:23

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

from ..detect_libs.deepsortDetection import DeepSortDetection

from ..deep_sort_libs.utils.parser import get_config
from ..deep_sort_libs.deep_sort.deep.feature_extractor import Extractor
from ..deep_sort_libs.deep_sort.sort.pose_tracker import pDetection, pTracker
from ..deep_sort_libs.deep_sort.sort.detection import Detection as ds_detect
from ..deep_sort_libs.deep_sort.sort.preprocessing import non_max_suppression
from ..deep_sort_libs.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric

class PoseDeepSortDetection(DeepSortDetection):

    def __init__(self, args, objName, scriptName):
        super(PoseDeepSortDetection, self).__init__(args, objName, scriptName)
        self.tracker = pTracker(self.metric, max_iou_distance = self.max_iou_distance,
                               max_age = self.max_age, n_init = self.n_init, buffer=self.buffer)

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
        self.buffer           = int(self.cf.get(self.objName, 'buffer'))

    def update(self, bbox_xywh, kps, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)

        # Create Detections object.
        detections = [pDetection(box, np.concatenate((kp.numpy(), conf.numpy()), axis=1),
                                conf.mean().numpy(), ft) for box, kp, conf, ft in zip(bbox_xywh, kps, confidences, features)]
        # print(f'deepsort detection: {detections}')

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        boxes, ids, keypoints = [], [], []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            keypoint = track.keypoints_list

            boxes.append(np.array([x1, y1, x2, y2], dtype=np.int))
            ids.append(np.array([track_id], dtype=np.int))
            keypoints.append(keypoint)
            #outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(boxes) > 0:
            boxes = np.stack(boxes, axis=0)
            ids = np.stack(ids, axis=0)
            keypoints = np.stack(keypoints, axis=0)
        return boxes, ids, keypoints
