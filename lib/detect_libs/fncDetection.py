import os
import cv2
from ..detect_libs.fasterDetection import FasterDetection
from ..detect_utils.tryexcept import try_except
from ..detect_utils.utils import compute_iou
class fncDetection(FasterDetection):
    def __init__(self, args, objName,scriptName):
        super(fncDetection, self).__init__(args, objName,scriptName)

    @try_except()
    def saveImage(self,imgName,img):
        pass 


    @try_except()
    def postProcess(self,im,image_name,detectBoxes):
        self.log.line('postProcess')
        h,w,_=im.shape
        results = {}
        for dBox in detectBoxes:
            label = dBox[0]
            index = dBox[1]
            xmin = dBox[2]
            ymin = dBox[3]
            xmax = dBox[4]
            ymax = dBox[5]
            prob = dBox[6]
            if 'fnc' not in label:
                continue

            resultImg = im[ymin:ymax, xmin:xmax]
            imgnam,extension = os.path.splitext(image_name)
            resizedName = imgnam + "_" + label + "_resized_" + str(index)
            resizedFile = os.path.join(self.tmpPaths['fnc'] , resizedName + '.jpg') 
            cv2.imwrite(resizedFile,resultImg)
            print(resizedName)

            results[resizedName] = {'label':label,'index':index,'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax,'prob':prob}
        return results
