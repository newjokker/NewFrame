import os
import cv2
import json
import configparser
import numpy as np
import tensorflow as tf
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except,timeStamp
from ..detect_utils.utils import isBoxExist
from ..detect_utils.cryption import salt, decrypt_file
from ..detect_libs.abstractBase import detection
from ..faster_libs.model.nms_wrapper import nms
from ..faster_libs.nets.resnet_v1 import resnetv1
from ..faster_libs.model.config import cfg
from ..faster_libs.model.test import im_detect

class FasterDetection(detection):
    def __init__(self, args, objName, scriptName):
        super(FasterDetection, self).__init__(objName, scriptName)
        self.suffixs = [".meta", ".index", ".data-00000-of-00001"]
        self.filterFuncs = [self.checkVisible]

        self.readArgs(args)
        self.readCfg()
        self.log = detlog(self.modelName, self.objName, self.logID)

    def readArgs(self, args):
        self.portNum  = args.port
        self.gpuID    = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host     = args.host
        self.logID    = args.logID

    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName  = self.cf.get('common', 'model')
        self.encryption = self.cf.getboolean("common", 'encryption')
        self.debug      = self.cf.getboolean("common", 'debug')

        self.tfmodelName     = self.cf.get(self.objName,'modelName')
        #self.dataset         = self.cf.get(self.objName, 'dataset')
        #self.demonet         = self.cf.get(self.objName, 'net')
        self.SCALES          = (int(self.cf.get(self.objName, 'test_minsize')), )
        self.MAX_SIZE        = int(self.cf.get(self.objName, 'test_maxsize'))
        self.anchorScales    = json.loads(self.cf.get(self.objName, "anchorScales"))
        self.anchorRatios    = json.loads(self.cf.get(self.objName, "anchorRatios"))
        self.CLASSES         = tuple(self.cf.get(self.objName, 'classes').strip(',').split(','))
        self.VISIBLE_CLASSES = tuple(self.cf.get(self.objName, 'visible_classes').strip(',').split(','))
        self.nmsThresh       = self.cf.getfloat(self.objName, 'nms_threshold')
        self.confThresh      = self.cf.getfloat(self.objName, 'conf_threshold')
        self.iouThresh       = self.cf.getfloat(self.objName, 'iou_threshold')
        try:
            self.areaThresh = self.cf.getint(self.objName, 'detected_box_threshold')
        except:
            self.areaThresh = 0

    @try_except()
    def setCfg(self):
        cfg.TEST.HAS_RPN  = True  # Use RPN for proposals
        cfg.TEST.SCALES   = self.SCALES
        cfg.TEST.MAX_SIZE = self.MAX_SIZE

    @try_except()
    def model_restore(self):
        self.log.info('===== model restore start =====')
        self.setCfg()

        if self.encryption:
            tfmodel = self.dncryptionModel()
        else:
            tfmodel = os.path.join(self.modelPath,self.tfmodelName)
        if not os.path.isfile(tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                           'our server and place them properly?').format(tfmodel + '.meta'))

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        tfconfig.gpu_options.per_process_gpu_memory_fraction=self.gpuRatio
        tfconfig.gpu_options.visible_device_list=str(self.gpuID)
        self.sess = tf.Session(config=tfconfig)
        self.net = resnetv1(num_layers=101)
        self.net.create_architecture("TEST", len(self.CLASSES), tag='default',
                                     anchor_scales=self.anchorScales, anchor_ratios=self.anchorRatios)
        saver = tf.train.Saver()
        saver.restore(self.sess, tfmodel)   # 将参数 tfmodel 写入 sess中

        if self.encryption:
            self.delDncryptionModel() 
        self.warmUp()
        self.log.info('===== model restore end =====')

    @try_except()
    def dncryptionModel(self):
        if False == os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)

        ### 解密模型 ####
        name, ext = os.path.splitext(self.tfmodelName)
        for sf in self.suffixs:
            model_origin_name = name + ext + sf
            model_locked_name = name + "_locked" + ext + sf
            origin_Fmodel = os.path.join(self.cachePath, model_origin_name)
            locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
            decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        ### 解密后的模型 ####
        tfmodel = os.path.join(self.cachePath, self.tfmodelName)
        return tfmodel

    @try_except()
    def delDncryptionModel(self):
        ###删除模型 ####
        for sf in self.suffixs:
            model_origin_name = self.tfmodelName + sf
            origin_Fmodel = os.path.join(self.cachePath, model_origin_name)
            os.remove(origin_Fmodel)
        self.log.info('delete dncryption model successfully! ')
        return

    @try_except()
    def warmUp(self):
        im = 128 * np.ones((self.SCALES[0], self.MAX_SIZE, 3), dtype=np.uint8)
        self.detect(im, 'warmup.jpg')
    
    @timeStamp()
    @try_except()
    def detect(self, im, image_name="default.jpg", mirror=False):
        self.log.info('===== {} for {} ====='.format(self.modelName, image_name))
        if im is None:
            self.log.info("【Waring】{} == None !!!".format(image_name))
            return []

        else:
            h, w, _ = im.shape
            self.log.info('im_detect')
            scores, boxes = im_detect(self.sess, self.net, im)

            if mirror == True:
                mirror_im = cv2.flip(im,1)
                #im90=np.rot90(im)
                scoresM, boxesM = im_detect(self.sess, self.net, mirror_im)
                #scores,boxes = self.combine90(h,boxes0,scores0,boxesM,scoresM)
                scores , boxes = self.combineMirror(w, boxes, scores, boxesM, scoresM)

            tot_labels = self.boxes_select(im, scores, boxes)
            self.log.info('before filters:\n', tot_labels[1:])

            voc_labels = self.filters(tot_labels)
            self.log.info('after  filters:\n', voc_labels)
            return voc_labels

    @try_except()
    def filters(self, tot_label):
        voc_labels = []
        if len(tot_label) > 0:
            for i, obj in enumerate(tot_label):
                label, p, xmin, ymin, xmax, ymax = obj
                voc_labels.append([label.strip(), i, int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax)), float(p)])
        self.tot_labels = voc_labels

        if len(self.filterFuncs) > 0:
            for func in self.filterFuncs:
                voc_labels = func(voc_labels)
        return voc_labels

    @try_except()
    def checkVisible(self, tot_label):
        return [obj for obj in tot_label if obj[0] in self.VISIBLE_CLASSES]

    @try_except()
    def boxes_select(self, im, scores, boxes):
        tot_label=np.zeros([1,6]).astype(np.str)
        for cls_ind, cls in enumerate(self.CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, self.nmsThresh, self.gpuID)
            dets = dets[keep, :]
            Nsp , Sp_pos = self.vis_detections(im, cls, dets, thresh=self.confThresh)
            if Nsp != 0:
               tot_label=np.concatenate([tot_label,Sp_pos])
        return tot_label

    @try_except()
    def vis_detections(self,im, class_name, dets,thresh=0.5):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]

        if len(inds) == 0:
            return 0,['0','0']
        xiangzi=np.zeros([len(inds),4])
        labelT=np.zeros([len(inds),1]).astype(np.str)
        conf_lev=np.zeros([len(inds),1]).astype(np.str)
        boxList = []
        c = 0 #true box index 
        for i in inds:
            bbox = dets[i, :4]
            if(isBoxExist(boxList,(round(bbox[0]),round(bbox[1]),round(bbox[2]),round(bbox[3])),self.iouThresh)):
                continue
            else:
                boxList.append((round(bbox[0]),round(bbox[1]),round(bbox[2]),round(bbox[3])))
            score = dets[i, -1]
            xiangzi[c,0]=round(bbox[0].astype(int))    # XMIN
            xiangzi[c,1]=round(bbox[1].astype(int))    # YMIN
            xiangzi[c,2]=round(bbox[2].astype(int))    # XMAX
            xiangzi[c,3]=round(bbox[3].astype(int))    # YMAX 
            labelT[c,0]=class_name
            conf_lev[c,0]=str(round(score,2))
            #conf_lev[c,0]=str(score)
            c = c + 1
        trueXiangzi = xiangzi[0:c,]
        trueLabelT = labelT[0:c,]
        trueConfLev = conf_lev[0:c,]
        rets=np.concatenate([trueLabelT,  trueConfLev , trueXiangzi.astype(np.str)],axis=1)
        return len(inds) ,rets

    @try_except()
    def combineMirror(self,w,boxesO,scoresO,boxesM,scoresM):
        classNum = len(self.CLASSES)
        for i in range(classNum):
            xmax = w - boxesM[:,4*i]
            xmin = w - boxesM[:,4*i + 2]
            boxesM[:,4*i + 2] = xmax
            boxesM[:,4*i] = xmin
        scores = np.vstack((scoresO,scoresM))
        boxes = np.vstack((boxesO,boxesM))
        return scores,boxes
   
    @try_except() 
    def combine90(self,h,boxes0,scores0,boxes90,scores90):
        classNum = len(self.CLASSES)
        for i in range(classNum):
            xmin = boxes90[:,4*i]
            xmax = boxes90[:,4*i + 2]
            ymin = boxes90[:,4*i + 1]
            ymax = boxes90[:,4*i + 3]
            newXmin = ymin
            newXmax = ymax
            newYmin = h - xmax
            newYmax = h - xmin
            boxes90[:,4*i] = newXmin
            boxes90[:,4*i + 1] = newYmin
            boxes90[:,4*i + 2] = newXmax
            boxes90[:,4*i + 3] = newYmax
        scores = np.vstack((scores0,scores90))
        boxes = np.vstack((boxes0,boxes90))
        return scores,boxes

