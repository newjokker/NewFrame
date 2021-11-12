import numpy as np
import os
import cv2
from ..detect_utils.tryexcept import try_except
from .yolov5Detection import YOLOV5Detection

class yytDetection(YOLOV5Detection):
    def __init__(self, args, objName, scriptName):
        super(yytDetection, self).__init__(args, objName, scriptName)
        self.card_resized_img_path = self.getTmpPath('card')
        self.person_resized_img_path = self.getTmpPath('person')

    @try_except()
    def merge(self, im, name, boxes, classes, scores):
        img_h, img_w, _ = im.shape

        objects = self.post_process(boxes, classes, scores)
        results = []
        for i, dBox in enumerate(objects):
            label, index, xmin, ymin, xmax, ymax, p = dBox

            resized_name = "None"
            if label == "card" or label == "person":
                h, w = ymax - ymin + 1, xmax - xmin + 1
                result_img = im[ymin:ymax, xmin:xmax]
                resized_name = name + "_resized_" + self.objName + "_" + str(index) + "_" + label
                if label == "card":
                    cv2.imwrite(os.path.join(self.card_resized_img_path, resized_name + '.jpg'), result_img)
                else:
                    cv2.imwrite(os.path.join(self.person_resized_img_path, resized_name + '.jpg'), result_img)
            elif label == "cell":
                index = "cell"
            elif label == "chair":
                index = "chair"
            else:
                pass

            results.append({
                "resizedName":resized_name,
                "objName":self.objName,
                "label":label,
                "index":index,
                "bbox":[xmin, ymin, xmax, ymax]})
        return results
                                                               

