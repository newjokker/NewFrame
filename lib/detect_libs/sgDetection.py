import os
import cv2
import numpy as np
from ..detect_utils.utils import mat_inter, check_record_file_path
from ..detect_utils.tryexcept import try_except
from ..detect_libs.fasterDetection import FasterDetection


class SgDetection(FasterDetection):
    def __init__(self, args, objName, scriptName):
        super(SgDetection, self).__init__(args, objName, scriptName)
        self.resizedImgPath = os.path.join(self.tmpPath, 'ljc_demo', 'resizedImg')


    @try_except()
    def postProcess(self,name,ljcBox,detectBoxes,results):
        for dBox in detectBoxes:
            if name in ljcBox['resizedName']:
                label, index, xmin, ymin, xmax, ymax, p = dBox
                xmap, ymap = ljcBox['xmin'], ljcBox['ymin']

                xmin = xmin + xmap
                ymin = ymin + ymap
                xmax = xmax + xmap
                ymax = ymax + ymap
                results.append({'label':label, 'index':index, 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax, 'prob': p})
        return results
        
    @try_except()
    def checkOthersGlobal(self, kkxBoxes):
        try:
            b = []
            self.log.info('in checkOthersGlobal')
            otherList = [(0,0,0,0)]
            for kkxBox in kkxBoxes:
                xmin = float(kkxBox['xmin'])
                ymin = float(kkxBox['ymin'])
                xmax = float(kkxBox['xmax'])
                ymax = float(kkxBox['ymax'])
                if self.isBoxIn_other(otherList, (xmin,ymin,xmax,ymax)):
                    continue
                else:
                    otherList.append((xmin,ymin,xmax,ymax))
                    b.append(kkxBox)
            
            return b
        except Exception as e:
            self.log.info('GOT ERROR---->')
            self.log.info(e)
            self.log.info(e.__traceback__.tb_frame.f_globals["__file__"])
            self.log.info(e.__traceback__.tb_lineno)

            
    @try_except()
    def isBoxIn_other(self, boxList, bbox):
        for box in boxList:
            if (self.solve_coincide_other(box, bbox, 0.2)):
                return True
        return False


    @try_except()
    def solve_coincide_other(self, box1, box2, iouThresh):
        if mat_inter(box1, box2) == True:
            coincide = self.compute_iou_other(box1, box2)
            return coincide > iouThresh
        else:
            return False


    @try_except()
    def compute_iou_other(self, box1, box2):
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / min(area1, area2)
        return coincide



