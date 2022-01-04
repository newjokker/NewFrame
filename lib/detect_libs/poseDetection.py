# -*- coding: utf-8 -*-
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

from ..alphapose_libs.models import builder
from ..alphapose_libs.utils.config import update_config
from ..alphapose_libs.utils.detector import DetectionLoader
from ..alphapose_libs.utils.transforms import flip, flip_heatmap, get_func_heatmap_to_coord
from ..alphapose_libs.utils.vis import getTime
from ..alphapose_libs.utils.webcam_detector import WebCamDetectionLoader
from ..alphapose_libs.utils.presets import SimpleTransform
#
from ..JoTools.txkjRes.deteRes import DeteRes, DeteObj
from ..JoTools.txkjRes.pointRes import PointRes, PointObj


class PoseDetection(detection):

    def __init__(self, args, objName, scriptName):
        super(PoseDetection, self).__init__(objName, scriptName)

        self.readArgs(args)
        self.readCfg()
        self.device = torch.device("cuda:" + str(self.gpuID)) if self.gpuID else 'cpu'
        print(f'cuda device: {self.device}')
        self.log = detlog(self.modelName, self.objName, self.logID)

        self.cfg = update_config(self.pose_cfg)
        self._input_size = self.cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        self._sigma = self.cfg.DATA_PRESET.SIGMA
        self._norm_type = self.cfg.LOSS.get('NORM_TYPE', None)

        self.dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        if self.cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                self.dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.cfg)

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
        self.pose_cfg         = self.cf.get(self.objName, 'pose_cfg')
        self.batchsize   = int(self.cf.get(self.objName, 'batch_size'))
        self.flip             = self.cf.getboolean(self.objName, 'flip')

    @try_except()
    def model_restore(self):
        self.log.info('===== model restore start =====')
        # 加密模型
        if self.encryption:
            model_path = self.dncryptionModel()
        else:
            model_path = os.path.join(self.modelPath, self.model_path)
        self.log.info('model_path is:', model_path)

        # Load pose model
        self.model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)

        print('Loading pose model from %s...' % (model_path))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # print(next(model.parameters()).device)
        self.model.to(self.device)
        self.model.eval()

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
        # fixme 我跑的时候说是找不到图片看看是什么问题
        # os.remove(self.model_path)
        # self.log.info('delete dncryption model successfully! ')
        return

    @try_except()
    def warmUp(self):
        img = torch.zeros(self.batchsize, 3, *self._input_size).to(self.device)
        keypoints = self.model(img)
        print(f'model warm up ended, Let\'s start!')
        return

    @try_except()
    def detect(self, im, boxes, image_name="default.jpg"):
        if boxes is None:
            return None, None

        inps, cropped_boxes = self.preProcess(im, boxes, image_name)

        batchSize = self.batchsize
        if self.flip:
            batchSize = int(batchSize/2)

        inps = inps.to(self.device)
        datalen = inps.size(0)
        leftover = 0
        if (datalen) % batchSize:
            leftover = 1
        num_batches = datalen // batchSize + leftover
        # heat map
        hm = []
        for j in range(num_batches):
            inps_j = inps[j*batchSize:min((j+1)*batchSize, datalen)]
            if self.flip:
                inps_j = torch.cat((inps_j, flip(inps_j)))

            hm_j = self.model(inps_j)
            if self.flip:
                hm_j_flip = flip_heatmap(hm_j[int(len(hm_j)/2):], self.dataset.joint_pairs, shift=True)
                hm_j = (hm_j[0:int(len(hm_j)/2)]+hm_j_flip)/2
            hm.append(hm_j)
        hm = torch.cat(hm)
        hm = hm.cpu()
        hm = torch.cat([hm[:, :1, ...], hm[:, 5:, ...]], dim=1)
        #
        return hm, cropped_boxes

    @try_except()
    def detectSOUT(self, im, boxes, image_name="default.jpg"):
        # boxes 的格式是 [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
        inps, cropped_boxes = self.preProcess(im, boxes, image_name)

        batchSize = self.batchsize
        if self.flip:
            batchSize = int(batchSize/2)

        inps = inps.to(self.device)
        datalen = inps.size(0)
        leftover = 0
        if (datalen) % batchSize:
            leftover = 1
        num_batches = datalen // batchSize + leftover
        # heat map
        hm = []
        for j in range(num_batches):
            inps_j = inps[j*batchSize:min((j+1)*batchSize, datalen)]
            if self.flip:
                inps_j = torch.cat((inps_j, flip(inps_j)))

            hm_j = self.model(inps_j)
            if self.flip:
                hm_j_flip = flip_heatmap(hm_j[int(len(hm_j)/2):], self.dataset.joint_pairs, shift=True)
                hm_j = (hm_j[0:int(len(hm_j)/2)]+hm_j_flip)/2
            hm.append(hm_j)
        #
        hm = torch.cat(hm)
        hm = hm.cpu()
        hm = torch.cat([hm[:, :1, ...], hm[:, 5:, ...]], dim=1)
        #
        # todo ht, cropped_boxes 后面的操作接上
        point_res = self.postProcess(hm, cropped_boxes)
        return point_res

    @try_except()
    def postProcess_new(self, hm, cropped_boxes):

        # heatmap to coord in train.py
        if hm is None:
            return None, None

        assert hm.dim() == 4
        if hm.size()[1] == 136:
            eval_joints = [*range(0,136)]
        elif hm.size()[1] == 26:
            eval_joints = [*range(0,26)]
        elif hm.size()[1] == 13:
            eval_joints = [*range(0,13)]

        point_res = PointRes()
        group_id = 0
        #
        for i in range(hm.shape[0]):
            bbox = cropped_boxes[i].tolist()
            pose_coord, pose_score = self.heatmap_to_coord(hm[i][eval_joints], bbox, hm_shape=self._output_size, norm_type=self._norm_type)
            coord = pose_coord.tolist()
            score = pose_score.tolist()
            # fixme 遍历一个对象中的点，放到 pointRes 中
            group_id += 1
            for point_index in range(len(coord)):
                point_res.add_obj(x=coord[point_index][0], y=coord[point_index][1], tag='default', conf=score[point_index][0], assign_id=group_id)
        # print
        # point_res.print_as_fzc_format()
        return point_res

    @try_except()
    def postProcess(self, hm, cropped_boxes):

        # heatmap to coord in train.py
        if hm is None:
            return None, None

        assert hm.dim() == 4
        if hm.size()[1] == 136:
            eval_joints = [*range(0, 136)]
        elif hm.size()[1] == 26:
            eval_joints = [*range(0, 26)]
        elif hm.size()[1] == 13:
            eval_joints = [*range(0, 13)]
        # print(f'hm size: {hm.shape}')

        pose_coords = []
        pose_scores = []
        for i in range(hm.shape[0]):
            bbox = cropped_boxes[i].tolist()
            # print(bbox)
            # print("******bbox*****************")
            pose_coord, pose_score = self.heatmap_to_coord(
                hm[i][eval_joints], bbox, hm_shape=self._output_size, norm_type=self._norm_type)
            coord = pose_coord.tolist()
            score = pose_score.tolist()
            # print(type(pose_coord))
            # print(type(pose_score))
            # print("pose_coord:", coord)
            # print("pose_score:", score)
            # print("x:", int(coord[0][0]))
            # print("y:", int(coord[0][1]))
            # print("p:", score[0][0])
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            # print("pose_coord:", pose_coord)
            # print("&&&&&&&&&&&&coo")
            # print("pose_scores:", pose_scores)
        preds_kps = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)

        # print("preds_kps:", preds_kps)
        # print("preds_scores:", preds_scores)

        return preds_kps, preds_scores

    @try_except()
    def preProcess(self, im, boxes, image_name="default.jpg"):
        inps = torch.zeros(boxes.shape[0], 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.shape[0], 4)
        
        for i, box in enumerate(boxes):
            inps[i], cropped_box = self.transformation.test_transform(im, box)
            cropped_boxes[i] = torch.FloatTensor(cropped_box)
        return inps, cropped_boxes
