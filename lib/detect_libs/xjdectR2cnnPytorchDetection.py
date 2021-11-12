import os
import torch
from torchvision import transforms as T
import cv2,copy
import json
import math
import os, sys
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..','r2cnnPytorch_libs')
sys.path.insert(0, lib_path)
import configparser
import numpy as np
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except,timeStamp
from ..detect_utils.cryption import salt, decrypt_file
from ..detect_utils.utils import solve_coincide
from ..detect_libs.abstractBase import detection
from ..detect_libs.r2cnnPytorchDetection import R2cnnDetection
from lib.detect_utils.utils import polygon_inter
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.config import cfg


from Crypto.Cipher import AES
import struct
from xml.etree.ElementTree import Element,ElementTree
from xml.etree.ElementTree import tostring
from xml.etree.ElementTree import SubElement
import xml.etree.ElementTree as ET


class XjdectR2cnnDetection(R2cnnDetection):
    def __init__(self,args,objName,scriptName):
        super(XjdectR2cnnDetection, self).__init__(args,objName,scriptName)

    ####################################################
    # def postProcess_plus(self, im, name, detectBoxes):
    #     if detectBoxes == [] or len(detectBoxes) == 0:
    #         return []
    #
    #     ##基于线夹的外接矩形生成xj_cap
    #     xj_tgt_labels = ["XuanChuiXianJia", "YuJiaoShiXJ"]
    #     raw_h, raw_w, _ = im.shape
    #     index = 0
    #     xj_boxs = []
    #     extendRate = 0.2
    #
    #     xj_tmp_boxes = [p for p in detectBoxes if p[4] in xj_tgt_labels]
    #     other_boxes = [p for p in detectBoxes if p[4] not in xj_tgt_labels]
    #
    #     xj_det_boxes = []
    #     if len(xj_tmp_boxes) < 1:
    #         return []
    #     elif len(xj_tmp_boxes) > 1:
    #         #TODO:判断面积占比 小于0.75 保留大的
    #         print('判断面积占比 小于0.75 保留大的')
    #     else:
    #         xj_det_boxes = xj_tmp_boxes
    #
    #     ot_point1 = ot_point2 = xj_point1 = xj_point2 = 0
    #     if len(xj_det_boxes)>0:
    #         for ljj_box in xj_det_boxes:
    #             index += 1
    #             points = ljj_box[0:4]
    #             label = ljj_box[4]
    #             xmin, xmax, ymin, ymax = self.minirect2(points, raw_h, raw_w)
    #             xmin_new, xmax_new, ymin_new, ymax_new = self.flip_increase(xmin, xmax, ymin, ymax, raw_h, raw_w, extendRate)
    #             resizedName = name + "_resized_" + str(index)
    #             if len(other_boxes)>0:
    #                 for ot_box in other_boxes:
    #                     ot_points = ot_box[0:4]
    #                     iou = polygon_inter(ot_points,points)
    #                     print('^^^^ iou ={}',iou)
    #                     if iou>0:
    #                         ot_point1 = (int((ot_box[0][0]+ot_box[1][0])/2),int((ot_box[0][1]+ot_box[1][1])/2))
    #                         ot_point2 = (int((ot_box[2][0] + ot_box[2][0]) / 2), int((ot_box[3][1] + ot_box[3][1]) / 2))
    #
    #                         xj_point1 = (int((ljj_box[0][0]+ljj_box[3][0])/2),int((ljj_box[0][1]+ljj_box[3][1])/2))
    #                         xj_point2 = (int((ljj_box[1][0] + ljj_box[2][0]) / 2), int((ljj_box[1][1] + ljj_box[2][1]) / 2))
    #
    #             xj_boxs.append( {'resizedName': resizedName, 'label': label, 'index': index, 'xmin': xmin_new, 'ymin': ymin_new, 'xmax': xmax_new, 'ymax': ymax_new})
    #
    #
    #     if self.debug:
    #         pic_dir = os.path.join(self.tmpPath, self.scriptName, 'r2cnn')
    #         pic_path = os.path.join(pic_dir, name + '.jpg')
    #         if os.path.exists(pic_path):
    #             result_img = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), 1)
    #             for xjBox in xj_boxs:
    #                 label, index, xmin, xmax, ymin, ymax = xjBox['label'], xjBox['index'], xjBox['xmin'], xjBox['xmax'],  xjBox['ymin'], xjBox['ymax']
    #                 cv2.rectangle(result_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 140, 255), 3)
    #                 ###写字#####
    #                 font_style = cv2.FONT_HERSHEY_SIMPLEX  # 字体
    #                 font_size = 3
    #                 font_cuxi = 3
    #                 cv2.putText(result_img, str(label + '-' + str(index)), (xmin, ymin - 3), font_style, font_size,(0, 140, 255), font_cuxi)
    #
    #             cv2.line(result_img, ot_point1, ot_point2, (255, 0, 0), 6)
    #             cv2.line(result_img, xj_point1, xj_point2, (255, 255, 0), 6)
    #
    #             out_dir = os.path.join(self.tmpPath, self.scriptName, 'xj_tangel')
    #             os.makedirs(out_dir, exist_ok=True)
    #             out_path = os.path.join(out_dir, name + '.jpg')
    #             cv2.imwrite(out_path, result_img)
    #
    #             resizedImgPath = os.path.join(self.tmpPath, self.scriptName, 'resizedImg')
    #             os.makedirs(resizedImgPath, exist_ok=True)
    #             resultImg = im[ymin:ymax, xmin:xmax]
    #             imgpath = os.path.join(resizedImgPath, resizedName + '.jpg')
    #             cv2.imwrite(imgpath, resultImg)
    #
    #     return xj_boxs

    ####################################################

    def postProcess2(self, im, name, detectBoxes):

        if detectBoxes == [] or len(detectBoxes) == 0:
            return []

        ##基于线夹的外接矩形生成xj_cap
        xj_tgt_labels = ["XuanChuiXianJia","YuJiaoShiXJ"]
        exit_labels = ["LCLB","LKLB","LLLB","LCLB"]
        iou_labels = ["LSSXJ"]
        raw_h, raw_w, _ = im.shape
        index = 0
        xj_boxs= []
        other_boxes = []
        extendRate = 0.2

        for ljj_box in detectBoxes:
            points = ljj_box[0:4]
            label = ljj_box[4]
            xmin, xmax, ymin, ymax = self.minirect2(points, raw_h, raw_w)
            resizedName = ''
            if label in exit_labels:
                return []
            if label in iou_labels:
                other_boxes.append({'resizedName': resizedName, 'label':label,'index': index, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}) 
            if label in xj_tgt_labels:
                if ymin < 20 or ymax > raw_h -20 or xmin < 20 or xmax > raw_w - 20:
                    continue
                else:
                    index += 1
                xmin_new, xmax_new, ymin_new, ymax_new = self.flip_increase(xmin, xmax, ymin, ymax,raw_h, raw_w,extendRate)

                resizedName = name + "_resized_" + str(index)
                xj_boxs.append({'resizedName': resizedName, 'label':label,'index': index, 'xmin': xmin_new, 'ymin': ymin_new, 'xmax': xmax_new, 'ymax': ymax_new})
        remain_boxes = []
        for xj_box in xj_boxs:
            result = False
            box_xj = [xj_box['xmin'],xj_box['ymin'],xj_box['xmax'],xj_box['ymax']]
            for other_box in other_boxes:
                box_other = [other_box['xmin'],other_box['ymin'],other_box['xmax'],other_box['ymax']]
                result = solve_coincide(box_xj,box_other)
                if result:
                    self.log.info('xj box and other box is overlapping')
                    break
            if not result:
                self.log.info('got one')
                remain_boxes.append(xj_box)

        if self.debug:
            pic_dir = os.path.join(self.tmpPath, self.scriptName, 'r2cnn')
            pic_path = os.path.join(pic_dir,name+'.jpg')
            if os.path.exists(pic_path):
                result_img = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), 1)
                for xjBox in xj_boxs:
                    label, index,xmin, xmax, ymin, ymax = xjBox['label'],xjBox['index'],xjBox['xmin'],xjBox['xmax'],xjBox['ymin'],xjBox['ymax']
                    cv2.rectangle(result_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,140,255), 3)
                    ###写字#####
                    font_style = cv2.FONT_HERSHEY_SIMPLEX  # 字体
                    font_size = 3
                    font_cuxi = 3
                    cv2.putText(result_img, str(label+'-'+str(index)), (xmin, ymin - 3), font_style, font_size, (0,140,255), font_cuxi)

                out_dir = os.path.join(self.tmpPath,self.scriptName,'xj_tangel')
                os.makedirs(out_dir,exist_ok=True)
                out_path = os.path.join(out_dir, name+'.jpg')
                cv2.imwrite(out_path, result_img)

                resizedImgPath = os.path.join(self.tmpPath,self.scriptName,'resizedImg')
                os.makedirs(resizedImgPath, exist_ok=True)
                resultImg = im[ymin:ymax, xmin:xmax]
                imgpath = os.path.join(resizedImgPath, resizedName + '.jpg')
                cv2.imwrite(imgpath, resultImg)
        
        if len(remain_boxes) >= 3:
            return []
        return remain_boxes


    def postProcess3(self, im, name, detectBoxes,xj_tgt_labels,extendRate):

        if detectBoxes == [] or len(detectBoxes) == 0:
            return []

        ##基于线夹的外接矩形生成xj_cap

        raw_h, raw_w, _ = im.shape
        index = 0
        xj_boxs= []

        for ljj_box in detectBoxes:
            points = ljj_box[0:4]
            label = ljj_box[4]
            xmin, xmax, ymin, ymax = self.minirect2(points, raw_h, raw_w)
            resizedName = ''
            if label in xj_tgt_labels:
                index += 1
                xmin_new, xmax_new, ymin_new, ymax_new = self.flip_increase(xmin, xmax, ymin, ymax,raw_h, raw_w,extendRate)

                resizedName = name + "_resized_" + str(index)
                xj_boxs.append({'resizedName': resizedName, 'label':label,'index': index, 'xmin': xmin_new, 'ymin': ymin_new, 'xmax': xmax_new, 'ymax': ymax_new})

        if self.debug:
            pic_dir = os.path.join(self.tmpPath, self.scriptName, 'r2cnn')
            pic_path = os.path.join(pic_dir,name+'.jpg')
            if os.path.exists(pic_path):
                result_img = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), 1)
                for xjBox in xj_boxs:
                    label, index,xmin, xmax, ymin, ymax = xjBox['label'],xjBox['index'],xjBox['xmin'],xjBox['xmax'],xjBox['ymin'],xjBox['ymax']
                    cv2.rectangle(result_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,140,255), 3)
                    ###写字#####
                    font_style = cv2.FONT_HERSHEY_SIMPLEX  # 字体
                    font_size = 3
                    font_cuxi = 3
                    cv2.putText(result_img, str(label+'-'+str(index)), (xmin, ymin - 3), font_style, font_size, (0,140,255), font_cuxi)

                out_dir = os.path.join(self.tmpPath,self.scriptName,'xj_tangel')
                os.makedirs(out_dir,exist_ok=True)
                out_path = os.path.join(out_dir, name+'.jpg')
                cv2.imwrite(out_path, result_img)

                resizedImgPath = os.path.join(self.tmpPath,self.scriptName,'resizedImg')
                os.makedirs(resizedImgPath, exist_ok=True)
                resultImg = im[ymin:ymax, xmin:xmax]
                imgpath = os.path.join(resizedImgPath, resizedName + '.jpg')
                cv2.imwrite(imgpath, resultImg)

        return xj_boxs


    def flip_increase(self,xmin, xmax, ymin, ymax,raw_h, raw_w,extendRate=0):
        cap_x = xmax - xmin
        cap_y = ymax - ymin
        xmin_new = max(xmin - extendRate*cap_x,0)
        xmax_new = min(xmax + extendRate*cap_x,raw_w)
        ymin_new = max(ymin - cap_y -extendRate*cap_y,0)
        ymax_new = min(ymax + extendRate*cap_y,raw_h)

        return int(xmin_new),int(xmax_new),int(ymin_new),int(ymax_new)

    ####################################################
    '''
    use ljj_result to generate xj_cap by chengcheng
    '''
    ####################################################
    def postProcess(self, im, name, detectBoxes):

        if detectBoxes == [] or len(detectBoxes) == 0:
            return []
        xj_dict, BanHuanShuan_dict = self.generate_minRect_and_classify_boxes(im, detectBoxes)
        if xj_dict == {} or BanHuanShuan_dict == {}:
            return []

        in_xjs = self.select_box(xj_dict, BanHuanShuan_dict)
        for in_xj in in_xjs:
            if len(in_xj) == 1:
                return []

        xj_boxs = self.merger_xj_box(in_xjs, name, im)
        return xj_boxs


    def generate_minRect_and_classify_boxes(self, im, detectBoxes):
        raw_h, raw_w, _ = im.shape

        index = 0
        xj_dict = {}
        BanHuanShuan_dict = {}
        for ljj_box in detectBoxes:
            # point_1 = ljj_box[0]
            # point_2 = ljj_box[1]
            # point_3 = ljj_box[2]
            # point_4 = ljj_box[3]
            points = ljj_box[0:4]
            name = ljj_box[4]

            xmin, xmax, ymin, ymax = self.minirect2(points, raw_h, raw_w)


            if name == 'YuJiaoShiXJ' or name == 'XuanChuiXianJia':
                xj_dict[index] = [ymin, xmin, ymax, xmax]
                index += 1
            elif 'GuaBan' in name or 'GuaHuan' in name or 'LuoShuan' in name or 'WTGB' in name:
                BanHuanShuan_dict[index] = [ymin, xmin, ymax, xmax]
                index += 1
        return xj_dict, BanHuanShuan_dict

    def compute_iou(self, rec1, rec2):
        """
        computing IoU
        :param rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect)) * 1.0

    def select_box(self, xj_dict, BanHuanShuan_dict):
        in_xjs = []
        for xj_inx, xj in xj_dict.items():
            in_xj = {}
            tmp2 = copy.deepcopy(BanHuanShuan_dict)

            for BHS_inx, BHS in BanHuanShuan_dict.items():
                iou = self.compute_iou(xj, BHS)
                if iou > 0:
                    in_xj[BHS_inx] = BHS
                    if BHS_inx in tmp2.keys():
                        del tmp2[BHS_inx]
            in_xj_added = copy.deepcopy(in_xj)
            BanHuanShuan_dict = copy.deepcopy(tmp2)

            while (1):
                tmp1 = {}
                for ix, box in in_xj_added.items():
                    for BHS_inx, BHS in BanHuanShuan_dict.items():
                        iou = self.compute_iou(box, BHS)
                        if iou > 0:
                            tmp1[BHS_inx] = BHS
                            if BHS_inx in tmp2.keys():
                                del tmp2[BHS_inx]
                if tmp1 == {}:
                    break
                in_xj_added = copy.deepcopy(tmp1)
                BanHuanShuan_dict = copy.deepcopy(tmp2)
                in_xj = dict(list(in_xj.items()) + list(in_xj_added.items()))
            in_xj[xj_inx] = xj
            in_xjs.append(in_xj)
        return in_xjs

    def merger_xj_box(self, in_xjs, name, im):
        h, w, _ = im.shape
        index = 0
        xj_boxs = []
        for in_xj in in_xjs:
            boxes = []
            for ix, box in in_xj.items():
                boxes.append(box)
            xj_box = np.array(boxes)
            ymin = int(np.min(xj_box[:, 0], axis=0))
            xmin = int(np.min(xj_box[:, 1], axis=0))
            ymax = int(np.max(xj_box[:, 2], axis=0))
            xmax = int(np.max(xj_box[:, 3], axis=0))

            xmin = max(xmin - 30, 0)
            ymin = max(ymin - 20, 0)
            xmax = min(xmax + 30, w)
            ymax = min(ymax + 20, h)

            self.log.info('roi is:', [ymin, xmin, ymax, xmax])

            resizedName = name + "_resized_" + str(index)
            xj_boxs.append(
                {'resizedName': resizedName, 'index': index, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
            index += 1

            resizedImgPath = self.getTmpPath('resizedImg')
            if self.debug:
                resultImg = im[ymin:ymax, xmin:xmax]
                imgpath = os.path.join(resizedImgPath, resizedName + '.jpg')
                cv2.imwrite(imgpath, resultImg)

        return xj_boxs

