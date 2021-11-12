import os
import cv2
from ..detect_libs.fasterDetection import FasterDetection
from ..detect_utils.tryexcept import try_except

class jyzDetection(FasterDetection):
    def __init__(self, args, objName,scriptName):
        super(jyzDetection, self).__init__(args, objName,scriptName)

    @try_except()
    def postProcess(self,im,detectBoxes):
        results = []
        h,w,_ = im.shape
        for dBox in detectBoxes:
            label = dBox[0]
            index = dBox[1]
            xmin = max(dBox[2] - 50,0)
            ymin = max(dBox[3] - 50,0)
            xmax = min(dBox[4] + 50,w)
            ymax = min(dBox[5] + 50,h)
            results.append({'label':label,'index':index,'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax})
        return results

    @try_except()
    def postProcess2(self,im,name,detectBoxes):
        w = im.shape[1]
        h = im.shape[0]
        results = []
        for dBox in detectBoxes:
            label = dBox[0]
            index = dBox[1]
            xmin = max(dBox[2] - 150,0)
            ymin = max(dBox[3] - 150,0)
            xmax = min(dBox[4] + 150,w)
            ymax = min(dBox[5] + 150,h)
            resizedName = name +"_resized_"+str(index)
            resizedFile = os.path.join(self.getTmpPath('resizedImg'),resizedName+'.jpg')
            resultImg = im[ymin:ymax, xmin:xmax]
            results.append({'resizedName':resizedName,'label':label,'index':index,'xmin':xmin,'ymin':ymin})
            cv2.imwrite(resizedFile, resultImg)

        return results
 
