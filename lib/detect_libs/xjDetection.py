import os
import cv2
from ..detect_libs.fasterDetection import FasterDetection
from ..detect_utils.tryexcept import try_except
from ..detect_utils.utils import compute_iou
class xjDetection(FasterDetection):
    def __init__(self, args, objName,scriptName):
        super().__init__(args, objName,scriptName)


    @try_except()
    def postProcess(self,im, name, detectBoxes):
        results = []
        h,w,_ = im.shape
        resizedImgPath = self.getTmpPath('resizedImg')
        for dBox in detectBoxes:
            label = dBox[0]
            index = dBox[1]
            xmin = max(dBox[2] - 20,0)
            ymin = max(dBox[3] - 20,0)
            xmax = min(dBox[4] + 20,w)
            ymax = min(dBox[5] + 20,h)
            resizedName = name + "_resized_" + str(index)
            results.append({'resizedName':resizedName,'label':label,'index':index,'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax})
            if self.debug:
                resultImg = im[ymin:ymax,xmin:xmax]
                self.log.info('#################################')
                self.log.info(resizedImgPath,resizedName + '.jpg')
                imgpath = os.path.join(resizedImgPath, resizedName+'.jpg')
                cv2.imwrite(imgpath, resultImg)
        return results

