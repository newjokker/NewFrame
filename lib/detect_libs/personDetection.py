import os
import cv2
from scripts.drawbox import personFilter
from ..detect_utils.tryexcept import try_except
from ..detect_libs.fasterDetection import FasterDetection
this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PersonDetection(FasterDetection):
    def __init__(self, args, objName, scriptName):
        super(PersonDetection, self).__init__(args, objName, scriptName)
        self.person_resized_img_path = os.path.join(self.tmpPath, "yyt_demo", "person")

    @try_except()
    def post_process(self, im, name, detectBoxes):
        img_h, img_w, _ = im.shape

        results = []
        for i, dBox in enumerate(detectBoxes):
            label, index, xmin, ymin, xmax, ymax = dBox[:-1]

            if personFilter((img_h, img_w), xmin, ymin, xmax, ymax):
                result_img = im[ymin:ymax, xmin:xmax]
                resized_name = name + "_resized_" + self.objName + "_" + str(index) + "_" + label
                cv2.imwrite(os.path.join(self.person_resized_img_path, resized_name + '.jpg'), result_img)
                results.append({'resizedName': resized_name, 'objName': self.objName, 'label': label, 'index': index, 'bbox': [xmin, ymin, xmax, ymax]})
            else:
                self.log.info("delete person {}: {}".format(i, dBox))

        return results
