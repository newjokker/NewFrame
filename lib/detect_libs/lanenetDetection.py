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
from ast import literal_eval
from ..lanenet_libs import lanenet_postprocess
from ..lanenet_libs import lanenet

class Config(dict):
    def __init__(self,*args, **kwargs):
        if 'config_path' in kwargs:
            config_content = self._load_config_file(kwargs['config_path'])
            super(Config, self).__init__(config_content)
        else:
            super(Config, self).__init__(*args, **kwargs)
        self.immutable = False
                                                                    


    def __setattr__(self, key, value, create_if_not_exist=True):
        """
        :param key:
        :param value:
        :param create_if_not_exist:
        :return:
        """
        if key in ["immutable"]:
            self.__dict__[key] = value
            return

        t = self
        keylist = key.split(".")
        for k in keylist[:-1]:
            t = t.__getattr__(k, create_if_not_exist)
        t.__getattr__(keylist[-1], create_if_not_exist)
        t[keylist[-1]] = value
    
    def __getattr__(self, key, create_if_not_exist=True):
        """
        :param key:
        :param create_if_not_exist:
        :return:
        """
        if key in ["immutable"]:
            #print(key)
            return self.__dict__[key]
        if key not in self:
            if not create_if_not_exist:
                raise KeyError
            self[key] = Config()
        if isinstance(self[key], dict):
            self[key] = Config(self[key])
        return self[key]

    def __setitem__(self, key, value):
        """
        :param key:
        :param value:
        :return:
        """
        if self.immutable:
            raise AttributeError(
                 'Attempted to set "{}" to "{}", but SegConfig is immutable'.
                 format(key, value))

        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
            except SyntaxError:
                pass
        super(Config, self).__setitem__(key, value)

    def set_immutable(self, immutable):
        self.immutable = immutable
        for value in self.values():
            if isinstance(value, Config):
                value.set_immutable(immutable)
                                                                                                     
    def is_immutable(self):
        return self.immutable
            
class lanenetDetection(detection):
    def __init__(self, args, objName, scriptName):
        super(lanenetDetection, self).__init__(objName, scriptName)
        self.suffixs = [".meta", ".index", ".data-00000-of-00001"]
        self.filterFuncs = []

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
        self.cfg = configparser.ConfigParser()
        self.cfg.read(self.cfgPath)
        self.modelName  = self.cfg.get('common', 'model')
        self.encryption = self.cfg.getboolean("common", 'encryption')
        self.debug      = self.cfg.getboolean("common", 'debug')

        self.lanenetCfg = Config()
        self.tfmodelName = self.cfg.get(self.objName,'modelName')
        self.lanenetCfg.MOVING_AVE_DECAY = self.cfg.getfloat(self.objName,'MOVING_AVE_DECAY')
        self.lanenetCfg.EMBEDDING_FEATS_DIMS = self.cfg.getint(self.objName,'EMBEDDING_FEATS_DIMS')
        self.lanenetCfg.FRONT_END = self.cfg.get(self.objName,'FRONT_END')
        self.lanenetCfg.DBSCAN_EPS = self.cfg.getfloat(self.objName,'DBSCAN_EPS')
        self.lanenetCfg.DBSCAN_MIN_SAMPLES = self.cfg.getint(self.objName,'DBSCAN_MIN_SAMPLES')
        self.lanenetCfg.WEIGHT_DECAY = self.cfg.getfloat(self.objName,'WEIGHT_DECAY')
        self.lanenetCfg.OHEM_ENABLE = False#self.cfg.get(self.objName,'OHEM_ENABLE')
        self.lanenetCfg.OHEM_SCORE_THRESH = self.cfg.getfloat(self.objName,'OHEM_SCORE_THRESH')
        self.lanenetCfg.OHEM_MIN_SAMPLE_NUMS = self.cfg.getint(self.objName,'OHEM_MIN_SAMPLE_NUMS')
        self.lanenetCfg.BISENETV2_GE_EXPAND_RATIO = self.cfg.getint(self.objName,'BISENETV2_GE_EXPAND_RATIO')
        self.lanenetCfg.BISENETV2_SEMANTIC_CHANNEL_LAMBDA = self.cfg.getfloat(self.objName,'BISENETV2_SEMANTIC_CHANNEL_LAMBDA')
        self.lanenetCfg.BISENETV2_SEGHEAD_CHANNEL_EXPAND_RATIO = self.cfg.getint(self.objName,'BISENETV2_SEGHEAD_CHANNEL_EXPAND_RATIO')
        self.SPLIT_STEP = self.cfg.getint(self.objName,'point_split_step')
        self.CLASSES         = tuple(self.cfg.get(self.objName, 'classes').replace(" ", "").split(','))
        self.VISIBLE_CLASSES = tuple(self.cfg.get(self.objName, 'visible_classes').replace(" ", "").split(','))
        self.lanenetCfg.NUM_CLASSES = len(self.CLASSES) + 1
        self.lanenetCfg.LOSS_TYPE = self.cfg.get(self.objName,'LOSS_TYPE')
        #print(self.lanenetCfg)

    @try_except()
    def model_restore(self):
        self.log.info('===== model restore start =====')

        if self.encryption:
            tfmodel = self.dncryptionModel()
        else:
            tfmodel = os.path.join(self.modelPath,self.tfmodelName)
        if not os.path.isfile(tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                           'our server and place them properly?').format(tfmodel + '.meta'))
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        self.net = lanenet.LaneNet(phase='test', cfg=self.lanenetCfg)
        self.log.info('0')
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='LaneNet')
        self.log.info('0.5')
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=self.lanenetCfg)
        self.log.info('1')

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        tfconfig.gpu_options.allocator_type = 'BFC'
        tfconfig.gpu_options.per_process_gpu_memory_fraction=self.gpuRatio
        tfconfig.gpu_options.visible_device_list=str(self.gpuID)
        self.sess = tf.Session(config=tfconfig)
        self.log.info('1.5')
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(self.lanenetCfg.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
        self.log.info('2')
        saver = tf.train.Saver(variables_to_restore)
        self.log.info('2.5')
        saver.restore(self.sess, tfmodel)   # 将参数 tfmodel 写入 sess中
        self.log.info('3')

        if self.encryption:
            self.delDncryptionModel() 
        #self.warmUp()
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
    
    @try_except()
    def preProcess(self,image):
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        return image

    @timeStamp()
    @try_except()
    def detect(self, im, image_name="default.jpg", mirror=False):
        self.log.info('===== {} for {} ====='.format(self.modelName, image_name))
        if im is None:
            self.log.info("【Waring】{} == None !!!".format(image_name))
            return []

        else:
            self.log.info('################')
            self.log.info(type(im))
            self.log.info(im.shape)
            originH,originW,_ = im.shape
            im = self.preProcess(im)
            h, w, _ = im.shape
            self.log.info('im_detect')
            binary_seg_image, instance_seg_image = self.sess.run([self.binary_seg_ret, self.instance_seg_ret],feed_dict={self.input_tensor: [im]})

            postprocess_result = self.postProcess(binary_seg_image[0],instance_seg_image[0],im)
            lane_coords = []
            ratioH = originH/h
            ratioW = originW/w
            for i in range(postprocess_result.shape[0]):
                lane_coords.append((int(postprocess_result[i][0]*ratioW),int(postprocess_result[i][1]*ratioH)))

            #result = binary_seg_image[]
             
            #if mirror == True:
            #    mirror_im = cv2.flip(im,1)
            #    scoresM, boxesM = im_detect(self.sess, self.net, mirror_im)
            #    scores , boxes = self.combineMirror(w, boxes, scores, boxesM, scoresM)
            return lane_coords
      
    @try_except()
    def postProcess(self,binary_seg_image,instance_seg_image,im):
        lane_coords = self.postprocessor.postprocess(binary_seg_result=binary_seg_image,instance_seg_result=instance_seg_image,source_image=im)
        #coord_list = self.split_line(lane_coords)
        return lane_coords
        
    #@try_except()
    #def 

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
   
