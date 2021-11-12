import os
import cv2
from scripts.drawbox import personFilter
from ..detect_utils.tryexcept import try_except
from ..detect_libs.fasterDetection import FasterDetection
this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DoorDetection(FasterDetection):
    def __init__(self, args, objName, scriptName):
        super(DoorDetection, self).__init__(args, objName, scriptName)
        self.door_resized_img_path = os.path.join(self.tmpPath, "opendoor_demo", "door")
        os.makedirs(self.door_resized_img_path,exist_ok=True)

    @try_except()
    def post_process(self, im, name, detectBoxes):
        img_h, img_w, _ = im.shape

        results = []
        for i, dBox in enumerate(detectBoxes):
            label, index, xmin, ymin, xmax, ymax, _ = dBox
            print("label",label)
            resized_name = "None"

            if label == 'door':
                h, w = ymax - ymin + 1, xmax - xmin + 1
                result_img = im[ymin:ymax, xmin:xmax]
                resized_name = name + "_resized_" + self.objName + "_" + str(index) + "_" + label

                cv2.imwrite(os.path.join(self.door_resized_img_path, resized_name + ".jpg"), result_img)

                results.append(
                    {'resizedName': resized_name, 'objName': self.objName, 'label': label, 'index': index,'bbox': [xmin, ymin, xmax, ymax]}
                )

        return results
