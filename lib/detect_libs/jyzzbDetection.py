import os
import cv2
from ..detect_libs.fasterDetection import FasterDetection
from ..detect_utils.tryexcept import try_except

class jyzzbDetection(FasterDetection):
    def __init__(self, args, objName,scriptName):
        super(jyzzbDetection, self).__init__(args, objName,scriptName)

