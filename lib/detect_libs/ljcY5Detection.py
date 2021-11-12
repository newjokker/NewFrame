import os
import cv2
from ..detect_utils.tryexcept import try_except
from .yolov5Detection import YOLOV5Detection


class LjcDetection(YOLOV5Detection):
    def __init__(self, args, objName, scriptName):
        super(LjcDetection, self).__init__(args, objName, scriptName)
        self.resizedImgPath = self.getTmpPath('resizedImg')

    @try_except()
    def postProcess(self, im, name, detectBoxes):
        results = []
        try:
            for dBox in detectBoxes:
                label, index, xmin, ymin, xmax, ymax, prob = dBox

                resizedName = name + "_resized_" + self.objName + "_" + str(index)
                xExtend, yExtend = self.getExtendForLjc(label, xmax - xmin, ymax - ymin)
                ymin, ymax, xmin, xmax = self.getResizedImgSize(im, xmin, xmax, ymin, ymax, xExtend, yExtend)
                resultImg = im[ymin:ymax, xmin:xmax]

                w = im.shape[1]
                h = im.shape[0]
                S_im = w * h
                S_ljc = (xmax - xmin) * (ymax - ymin)
                S_rate = S_ljc / S_im
                self.log.info("-----{} : {} : {}".format(resizedName, label, S_rate))

                ljc_Srate_threshold = 0
                if S_rate > ljc_Srate_threshold:
                    ##截ljc图###
                    cv2.imwrite(os.path.join(self.resizedImgPath, resizedName + '.jpg'), resultImg)
                    results.append({'resizedName': resizedName, 'label': label, 'index': index, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

        except Exception as e:
            self.log.info(e)
            self.log.info(e.__traceback__.tb_frame.f_globals["__file__"])
            self.log.info(e.__traceback__.tb_lineno)
        return results

    def postProcess2(self, im, boxes, classes, scores):
        results = []
        for i, box in enumerate(boxes):
            label = self.class_dict[classes[i]]
            xmin, ymin, xmax, ymax = boxes[i]

            obj_h, obj_w = ymax - ymin, xmax - xmin
            xExtend, yExtend = (150, 150 - (obj_h - obj_w) // 2) if "L88" in label and obj_h >= obj_w else (150, 150)
            ymin, ymax, xmin, xmax = self.getResizedImgSize(im, xmin, xmax, ymin, ymax, xExtend, yExtend)

            results.append([label, scores[i], [xmin, ymin, xmax, ymax]])

        return results


    @try_except()
    def getExtendForLjc(self, label, w, h):
        switchTable = {
            "L88": (150, 150)
        }
        if "L88" in label:
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
