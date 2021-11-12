import os
import cv2
import numpy as np
from ..detect_libs.fasterDetection import FasterDetection
from ..detect_utils.tryexcept import try_except

class ncDetection(FasterDetection):
    def __init__(self, args, objName,scriptName):
        super(ncDetection, self).__init__(args, objName,scriptName)

    @try_except()
    def postProcess(self,im,name,detectBoxes):
        resizedImgPath = self.getTmpPath('resizedImg')
        h,w,_ = im.shape
        results = []
        for dBox in detectBoxes:
            label = dBox[0]
            index = dBox[1]
            xmin = dBox[2]
            ymin = dBox[3]
            xmax = dBox[4]
            ymax = dBox[5]
            resultImg = im[ymin:ymax,xmin:xmax]
            hsv = cv2.cvtColor(resultImg, cv2.COLOR_BGR2HSV)
            # red filter
            lower_red1 = np.array([0, 43, 46])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([156, 43, 46])
            upper_red2 = np.array([180, 255, 255])
            maskred1 = cv2.inRange(hsv,lower_red1,upper_red1)
            maskred2 = cv2.inRange(hsv,lower_red2,upper_red2)
            mask = maskred1 + maskred2
            maskT = np.sum(np.sum(mask))
            ratio = maskT / (mask.shape[0]*mask.shape[1]*255.0)
            ratio = float('{:.2f}'.format(ratio))
            if ratio >= 0.5:
                continue
            # green filter
            lower_green = np.array([35, 43, 46])
            upper_green = np.array([77, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            maskT = np.sum(np.sum(mask))
            ratio = maskT / (mask.shape[0]*mask.shape[1]*255.0)
            ratio = float('{:.2f}'.format(ratio))
            if ratio >= 0.14:
                continue
            resizedName = name+'_resized_nc_'+str(index)
            results.append({'resizedName':resizedName,'label':label,'index':index,'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax})
            imgpath = os.path.join(resizedImgPath, resizedName+'.jpg')
            cv2.imwrite(imgpath, resultImg)
        return results

