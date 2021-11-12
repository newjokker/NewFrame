import os
import cv2
from ..detect_libs.fasterDetection import FasterDetection
from ..detect_utils.tryexcept import try_except
from ..detect_utils.utils import compute_iou
class waipoDetection(FasterDetection):
    def __init__(self, args, objName,scriptName):
        super(waipoDetection, self).__init__(args, objName,scriptName)


    @try_except()
    def postProcess(self,im,name,detectBoxes):
        results = []
        h,w,_ = im.shape
        for dBox in detectBoxes:
            label, index, xmin, ymin, xmax, ymax, p = dBox
            results.append({'label':label, 'index':index, 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax, 'prob': p})
        return results

