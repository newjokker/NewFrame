import os
import cv2
import json
import configparser
import numpy as np
import tensorflow as tf
from ..detect_utils.log import detlog
from ..detect_utils.timer import Timer
from ..detect_utils.tryexcept import try_except,timeStamp
from ..detect_utils.cryption import salt, decrypt_file
from ..detect_libs.abstractBase import detection
from ..r2cnn_libs.data.io.image_preprocess import short_side_resize_for_inference_data
from ..r2cnn_libs.configs import cfgs
from ..r2cnn_libs.networks import build_whole_network
from ..r2cnn_libs.help_utils.tools import *
from ..r2cnn_libs.box_utils import draw_box_in_img
from ..r2cnn_libs.box_utils import coordinate_convert
from Crypto.Cipher import AES
import struct

class R2cnnDetection(detection):
    def __init__(self,args,objName,scriptName):
        super(R2cnnDetection, self).__init__(objName,scriptName)
        self.suffixs = [".meta", ".index", ".data-00000-of-00001"]
        self.readCfg()
        self.args = args
        self.readArgs()
        self.log = detlog(self.modelName, self.objName, self.logID)
        self.allLabels = None

    def readArgs(self):
        self.portNum = self.args.port
        self.gpuID = self.args.gpuID
        self.gpuRatio = self.args.gpuRatio
        self.host = self.args.host
        self.logID = self.args.logID


    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.debug = self.cf.getboolean('common','debug')
        self.modelName = self.cf.get('common', 'model')
        #self.anchorScales = json.loads(self.cf.get(self.objName,"anchorScales"))
        #self.anchorRatios = json.loads(self.cf.get(self.objName,"anchorRatios"))
        self.CLASSES = tuple(self.cf.get(self.objName,'classes').strip(',').split(','))
        self.VISIBLE_CLASSES = tuple(self.cf.get(self.objName,'visible_classes').strip(',').split(','))
        self.INVISIBLE_CLASSES = tuple(set(self.CLASSES).difference(set(self.VISIBLE_CLASSES)))
        self.demonet = self.cf.get(self.objName,'net')
        self.dataset = self.cf.get(self.objName,'DATASET_NAME')
        self.tfmodelName = self.cf.get(self.objName,'modelName')
        self.encryption = self.cf.getboolean("common",'encryption')
        self.rpnThresh=self.cf.getfloat(self.objName,'RPN_NMS_IOU_THRESHOLD')
        self.IMG_SHORT_SIDE_LEN=self.cf.getint(self.objName,'IMG_SHORT_SIDE_LEN')
        self.Box_type=self.cf.get(self.objName,'Box_type')
    
    
    @try_except()
    def setCfg(self):
        self.log.info(self.gpuID)
        cfgs.GPU_ID = int(self.gpuID)
        cfgs.CLASS_NUM = self.CLASS_NUM
        cfgs.RPN_IOU_POSITIVE_THRESHOLD =self.RPN_IOU_POSITIVE_THRESHOLD
        cfgs.FAST_RCNN_R_NMS_IOU_THRESHOLD=self.FAST_RCNN_R_NMS_IOU_THRESHOLD
        cfgs.USE_DROPOUT=self.USE_DROPOUT
        cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS=self.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS
        
            
        
    def alter(file,old_str,new_str):
        n=len(old_str);m=len(new_str)
        assert n==m, "wrong in replacing string"
        with open(file, "r", encoding="utf-8") as f1,open("%s.bak" % file, "w", encoding="utf-8") as f2:
            for line in f1:
                if old_str in line:
                    for i,old in enumerate(old_str):
                        new=new_str[i]
                        line = line.replace(old, new)
                f2.write(line)
        os.remove(file)
        os.rename("%s.bak" % file, file)   

    @try_except()
    def dncryptionModel(self):
        if False == os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)
        ### 解密模型 ####
        name,ext = os.path.splitext(self.tfmodelName)
        self.log.info(name)
        self.log.info(ext)
        for sf in self.suffixs:
            model_origin_name = name + ext + sf
            model_locked_name = name + "_locked" + ext + sf
            origin_Fmodel = os.path.join(self.cachePath, model_origin_name)
            locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
            decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        ### 解密后的模型 ####
        self.log.info('in dncryptionModel')
        self.log.info(self.tfmodelName)
        tfmodel = os.path.join(self.cachePath, self.tfmodelName).replace('_locked','')
        return tfmodel

    @try_except()
    def model_restore(self):
        self.log.info('model restore start')
        #self.setCfg()
        if self.encryption:
            tfmodel = self.dncryptionModel()
        else:
            tfmodel = os.path.join(self.modelPath,self.tfmodelName)
        self.log.info('tfmodel')
        if not os.path.isfile(tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                           'our server and place them properly?').format(tfmodel + '.meta'))
        self.net = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,is_training=False)
        result_R2CNN={}
        # 1. preprocess img
        self.img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
        self.img_batch = tf.cast(self.img_plac, tf.float32)
        self.img_batch = self.img_batch - tf.constant(cfgs.PIXEL_MEAN)
        self.img_batch = short_side_resize_for_inference_data(img_tensor=self.img_batch,target_shortside_len=self.IMG_SHORT_SIDE_LEN)
        
        self.det_boxes_h, self.det_scores_h, self.det_category_h, \
        self.det_boxes_r, self.det_scores_r, self.det_category_r = \
        self.net.build_whole_detection_network(input_img_batch=self.img_batch,gtboxes_h_batch=None,gtboxes_r_batch=None,mask_batch=None,gpu_id=int(self.gpuID))
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        restorer  = self.net.get_restorer_my()

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        tfconfig.gpu_options.per_process_gpu_memory_fraction=self.gpuRatio
        tfconfig.gpu_options.visible_device_list=str(self.gpuID)
        # init session
        self.sess = tf.Session(config=tfconfig)
        self.sess.run(init_op)
        restorer.restore(self.sess, tfmodel)
        # load network
        self.warmUp()
        self.log.info('model restore end')
        if self.encryption:
            self.delDncryptionModel()
        return


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
        import cv2
        im = 128 * np.ones((800,1200,3),dtype=np.uint8)
        v = self.detect(im, 'warmup.jpg')
    
    @timeStamp()
    @try_except()
    def detect(self,im,image_name="default.jpg",mirror=False):
        self.log.info('=========================')
        self.log.info(self.modelName + ' detection start')
        if im is None:
            self.log.info("【Waring】:"+ image_name +"  == None !!!")
            return []
        raw_h, raw_w = im.shape[0], im.shape[1]
        resized_img, det_boxes_h_, det_scores_h_, det_category_h_, \
        det_boxes_r_, det_scores_r_, det_category_r_ = \
                self.sess.run(
                    [self.img_batch, self.det_boxes_h, self.det_scores_h, self.det_category_h,
                     self.det_boxes_r, self.det_scores_r, self.det_category_r],
                    feed_dict={self.img_plac: im}
                )
        #det_detections_h = draw_box_in_img.draw_box_cv(np.squeeze(resized_img, 0),
        #                                                   boxes=det_boxes_h_,
        #                                                   labels=det_category_h_,
        #                                                   scores=det_scores_h_)
        resized_h ,resized_w = resized_img.shape[1],resized_img.shape[2]
        rotate_box_pos = draw_box_in_img.get_rotate_box_info(np.squeeze(resized_img, 0),
                                                         boxes=det_boxes_r_,
                                                         labels=det_category_r_,
                                                         scores=det_scores_r_)
        h_ratio = raw_h / resized_h
        w_ratio = raw_w / resized_w
        results = []
        for pos in rotate_box_pos:
            results.append({'x1':int(pos['x1']*w_ratio),'y1':int(pos['y1']*h_ratio),'x2':int(pos['x2']*w_ratio),'y2':int(pos['y2']*h_ratio)})
        self.log.info(results)
        if self.debug:
            rotate_image_path = self.getTmpPath('rotate')
            rotate_image,_ = draw_box_in_img.draw_rotate_box_cv(np.squeeze(resized_img, 0),
                                                         boxes=det_boxes_r_,
                                                         labels=det_category_r_,
                                                         scores=det_scores_r_)
            save_path = os.path.join(rotate_image_path,image_name)
            cv2.imencode('.jpg', rotate_image)[1].tofile(save_path)
        return results

    
def delete_scrfile(srcfile_path):
    os.remove(srcfile_path)
