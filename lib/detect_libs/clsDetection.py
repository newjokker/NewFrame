import os
import cv2
import configparser
import numpy as np
import tensorflow as tf
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import *
from ..detect_libs.abstractBase import detection
from ..detect_utils.cryption import decrypt_file, salt
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K

class ClsDetection(detection):
    def __init__(self, args, objName,scriptName):
        super(ClsDetection, self).__init__(objName,scriptName)
        self.objName = objName
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
        self.modelName   = self.cf.get('common', 'model')
        self.encryption  = self.cf.getboolean("common", 'encryption')
        self.debug       = self.cf.getboolean("common", 'debug')
        self.tfmodelName = self.cf.get(self.objName,'modelName')
        self.normSize    = self.cf.getint(self.objName, 'norm_size')
        self.confThresh  = self.cf.getfloat(self.objName,'conf_threshold')

    @try_except()
    def model_restore(self):
        self.graph = tf.get_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpuRatio
        config.gpu_options.visible_device_list = str(self.gpuID)
        set_session(tf.Session(config=config))

        if self.encryption:
            tfmodel = self.dncryptionModel()
        else:
            tfmodel = os.path.join(self.modelPath, self.tfmodelName)

        self.log.info(tfmodel)
        self.model = load_model(tfmodel,custom_objects={'tf': tf})
        self.warmUp()
        if self.encryption:
            os.remove(tfmodel)
            self.log.info('delete dncryption model successfully! ')

    @try_except()
    def dncryptionModel(self):
        if False == os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)

        ### 解密模型 ####
        name, ext = os.path.splitext(self.tfmodelName)
        model_locked_name = name + "_locked" + ext
        origin_Fmodel = os.path.join(self.cachePath, self.tfmodelName)
        locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
        decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        return origin_Fmodel

    @try_except()
    def warmUp(self):
        im = 128 * np.ones((self.normSize, self.normSize, 3),dtype=np.uint8)
        self.detect(im, 'warmup.jpg')

    @timeStamp()
    @try_except()
    def detect(self, im, image_name="default.jpg"):
        if im is None:
            self.log.info("【Waring】:"+ image_name +"  == None !!!")
            return None, 0
        else:
            image = cv2.resize(im, (self.normSize, self.normSize))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # classify the input image
            with self.graph.as_default():
                result = self.model.predict(image)[0]
            #self.log.info(result)
            proba = np.max(result)
            label = str(np.where(result==proba)[0])
            return label[1:-1], proba












