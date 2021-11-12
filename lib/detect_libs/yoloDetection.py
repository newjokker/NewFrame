#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import json
import configparser
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw
from ..yolo_libs.model import yolo_eval
from ..yolo_libs.utils import letterbox_image
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except,timeStamp
from ..detect_libs.abstractBase import detection

class YOLODetection(detection):
    def __init__(self,args,objName,scriptName):
        super(YOLODetection, self).__init__(objName,scriptName)
        self.readArgs(args)
        self.readCfg()
        self.log = detlog(self.modelName, self.objName, self.logID)

    def readArgs(self,args):
        self.portNum = args.port
        self.gpuID = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host = args.host
        self.logID = args.logID
        
    @try_except()
    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName = self.cf.get('common', 'model')
        self.encryption = self.cf.getboolean("common",'encryption')
        self.debug      = self.cf.getboolean("common", 'debug')
     
        self.model_path      = self.cf.get(self.objName,'modelName')
        self.anchors_path    = self.cf.get(self.objName,'anchors')
        
        self.classes_path    = self.cf.get(self.objName,'classes')
        self.VISIBLE_CLASSES = self.cf.get(self.objName, 'visible_classes')
        self.nmsThresh       = self.cf.getfloat(self.objName, 'nms_threshold')
        self.score           = self.cf.getfloat(self.objName, 'conf_threshold')
        self.iou             = self.cf.getfloat(self.objName, 'iou_threshold')
        self.model_image_size = tuple(json.loads(self.cf.get(self.objName, 'model_image_size')))
        
    @try_except()
    def model_restore(self):
        self.log.info('===== model restore start =====')
        
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.

        config.gpu_options.per_process_gpu_memory_fraction = self.gpuRatio
        config.gpu_options.visible_device_list=str(self.gpuID)

        K.set_session(tf.Session(config=config))

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        
        self.is_fixed_size = self.model_image_size != (None, None)

        self.boxes, self.scores, self.classes = self.generate()
        self.graph = tf.get_default_graph()
        #self.warmUp()

    @try_except()
    def _get_class(self):
        classes_path = os.path.join(self.modelPath,self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        print(class_names)
        return class_names
        
    @try_except()
    def _get_anchors(self):
        anchors_path = os.path.join(self.modelPath,self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        print(anchors)
        return anchors

    @try_except()
    def generate(self):
        model_path = os.path.join(self.modelPath,self.model_path)

        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        self.yolo_model = load_model(model_path, compile=False)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    @try_except()
    def detect(self, image,name):
        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        
        with self.graph.as_default():
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
        return_boxs = []
        return_class_name = []
        person_counter = 0
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            

            person_counter += 1
            
            box = out_boxes[i]

            xmin = int(box[1])
            ymin = int(box[0])
            xmax = int(box[3])
            ymax = int(box[2])
            p = 0.9
            return_boxs.append([predicted_class.strip(), i, int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax)), float(p)])
            
        return return_boxs

    def postProcess(self,im,name,detectBoxes):
        results = []
        h,w,_ = im.shape
        for dBox in detectBoxes:
            label, index, xmin, ymin, xmax, ymax, p = dBox
            results.append({'label':label, 'index':index, 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax, 'prob': p})
        return results
        
    def close_session(self):
        self.sess.close()
