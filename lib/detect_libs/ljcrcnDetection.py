import os
import cv2
from ..detect_utils.tryexcept import try_except
from ..detect_libs.r2cnnDetection import R2cnnDetection
this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class LjcrcnDetection(R2cnnDetection):
    def __init__(self,args, objName, scriptName):
        super(LjcrcnDetection, self).__init__(args, objName, scriptName)
        self.resizedImgPath = self.getTmpPath('resizedImg')

    @try_except()
    def postProcess(self, im, name, detectResults):
        results = []
        w = im.shape[1]
        h = im.shape[0]
        try:
            for detect in detectResults:
                print('^'*20)
                print(detect)
                boxes = detect['boxes']
                labels = detect['labels']
                for index,(box,label) in enumerate(list(zip(boxes,labels))):
                    resizedName = name +'_' + label + '_' + str(index)
                    xmin,xmax,ymin,ymax = box
                    xExtend, yExtend = self.getExtendForLjc(label, xmax - xmin, ymax - ymin)
                    ymin, ymax, xmin, xmax = self.getResizedImgSize(im, xmin, xmax, ymin, ymax, xExtend, yExtend)
                    ww = xmax - xmin
                    hh = ymax - ymin

                    xmintmp = max(xmin - ww, 1)
                    xmaxtmp = min(xmax + ww, w)
                    ymintmp = max(ymin - hh, 1)
                    ymaxtmp = min(ymax + hh, h)
                    resultImg = im[int(ymintmp):int(ymaxtmp), int(xmintmp):int(xmaxtmp)]

                    if label in ['ZGuaBan','YuJiaoShiXJ','XuanChuiXianJia','Sanjiaoban','UGuaHuan','XinXingHuan','TXXJ','LSXXJ','XieXingXJ', 'BGXJ', 'SXJ']:
                        results.append({'resizedName': resizedName, 'label': label, 'index': index, 'xmin': xmintmp, 'ymin': ymintmp,'xmax': xmaxtmp, 'ymax': ymaxtmp})
                        cv2.imwrite(os.path.join(self.resizedImgPath, name + '_' + str(label) + '_' + str(index) + '.jpg'),resultImg)
        except Exception as e:
            self.log.info(e)
            self.log.info(e.__traceback__.tb_frame.f_globals["__file__"])
            self.log.info(e.__traceback__.tb_lineno)
            print(e)
        print(results)
        return results

    @try_except()
    def getExtendForLjc(self, label, w, h):
        switchTable = {
            "Zhongchui": (150, 150)
        }
        if "Zhongchui" in label:
            if w > h:
                gap = (w - h) // 2
                return (150, 150)
            else:
                gap = (h - w) // 2
                return (150, 150 - gap)
        if label in switchTable.keys():
            return switchTable[label]
        else:
            return (150, 150)

    @try_except()
    def getResizedImgSize(self, im, xmin, xmax, ymin, ymax, xExtend, yExtend):
        width = im.shape[1]
        height = im.shape[0]
        xmin_r = max(xmin - xExtend, 0)
        xmax_r = min(xmax + xExtend, width)
        ymin_r = max(ymin - yExtend, 0)
        ymax_r = min(ymax + yExtend, height)
        return ymin_r, ymax_r, xmin_r, xmax_r
