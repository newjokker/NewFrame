import os
import cv2
from ..detect_libs.fasterDetection import FasterDetection
from ..detect_utils.tryexcept import try_except
from ..detect_utils.utils import compute_iou
class jyhDetection(FasterDetection):
    def __init__(self, args, objName,scriptName):
        super(jyhDetection, self).__init__(args, objName,scriptName)

    @try_except()
    def saveImage(self,imgName,img):
        pass 


    @try_except()
    def postProcess(self,im,image_name,detectBoxes,jyzBoxes):
        self.log.line('postProcess')
        self.log.line('jyzBoxes')
        self.log.info(jyzBoxes)
        h,w,_=im.shape
        results = {}
        for dBox in detectBoxes:
            label = dBox[0]
            index = dBox[1]
            xmin = max(dBox[2] - 10,1)
            ymin = max(dBox[3] - 10,1)
            xmax = min(dBox[4] + 10,w)
            ymax = min(dBox[5] + 10,h)
            jyhBox = [xmin,ymin,xmax,ymax]
            is_existed=False
            for jyzBox in jyzBoxes:
                self.log.info(jyhBox)
                box = [jyzBox['xmin'],jyzBox['ymin'],jyzBox['xmax'],jyzBox['ymax']]
                self.log.info(box)
                result=compute_iou(jyhBox,box)
                self.log.info("iou:",result)
                if result > 0:
                    is_existed=True
                    break
            if not is_existed:
                continue
            if 'upring' not in label and 'downring' not in label:
                continue
            resultImg = im[ymin:ymax, xmin:xmax]
            resizedName = image_name + "_" + label + "_resized_" + str(index)
            resizedFile = os.path.join(self.tmpPaths['jyh'] , resizedName + '.jpg') 
            cv2.imwrite(resizedFile,resultImg)
            resizedImg_long = None
            if label=='upring':
                resizedImg_long=im[dBox[3]:min(dBox[3]+dBox[5]-dBox[3]+dBox[5]-dBox[3],h),dBox[2]:dBox[4]]
                W_add=dBox[2]
                H_add=dBox[3]
            else:
                resizedImg_long=im[max(dBox[3]+dBox[3]-dBox[5],0):dBox[5],dBox[2]:dBox[4]]
                W_add=dBox[2]
                H_add=max(dBox[3]+dBox[3]-dBox[5],0)
            resizedName_long = image_name + "_" + label + "_resized_" + str(index)
            resizedFile_long = os.path.join(self.tmpPaths['jyh_long'], resizedName_long + '.jpg')
            cv2.imwrite(resizedFile_long,resizedImg_long)

            results[resizedName] = {'label':label,'index':index,'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax,'W_add':W_add,'H_add':H_add}
        return results

