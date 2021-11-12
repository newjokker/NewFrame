import os
import cv2
import json
import configparser
import numpy as np
import tensorflow as tf
from ..detect_utils.log import detlog
from ..detect_utils.timer import Timer
from ..detect_utils.tryexcept import try_except,timeStamp
from ..detect_libs.r2cnnDetection import R2cnnDetection
from ..r2cnn_libs.data.io.image_preprocess import short_side_resize_for_inference_data
from ..r2cnn_libs.configs import cfgs
from ..r2cnn_libs.networks import build_whole_network
from ..r2cnn_libs.help_utils.tools import *
from ..r2cnn_libs.box_utils import draw_box_in_img
from ..r2cnn_libs.box_utils import coordinate_convert

from Crypto.Cipher import AES
import struct

class jyzR2cnnDetection(R2cnnDetection):
    def __init__(self,args,objName,scriptName):
        super(jyzR2cnnDetection, self).__init__(args,objName,scriptName)

   
    #def postProcess(self,image_name,im):
