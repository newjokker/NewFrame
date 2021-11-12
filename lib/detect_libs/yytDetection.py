import os
import cv2
from scripts.drawbox import personFilter
from ..detect_utils.tryexcept import try_except
from ..detect_libs.fasterDetection import FasterDetection
this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class YytDetection(FasterDetection):
    def __init__(self, args, objName, scriptName):
        super(YytDetection, self).__init__(args, objName, scriptName)
        self.card_resized_img_path = self.getTmpPath('card')
        self.person_resized_img_path = self.getTmpPath('person')

    @try_except()
    def post_process(self, im, name, detectBoxes):
        img_h, img_w, _ = im.shape

        results = []
        for i, dBox in enumerate(detectBoxes):
            label, index, xmin, ymin, xmax, ymax, p = dBox
            resized_name = "None"

            if label == "card" or label == "person":
                h, w = ymax - ymin + 1, xmax - xmin + 1
                result_img = im[ymin:ymax, xmin:xmax]
                resized_name = name + "_resized_" + self.objName + "_" + str(index)+ "_" + label

                if label == "card":
                    if p >= 0.8:
                        cv2.imwrite(os.path.join(self.card_resized_img_path, resized_name + '.jpg'), result_img)
                    else:
                        continue
                else:
                    #if personFilter((img_h, img_w), xmin, ymin, xmax, ymax):
                    if p >= 0.85:
                        cv2.imwrite(os.path.join(self.person_resized_img_path, resized_name + '.jpg'), result_img)
                    # else:
                    #     self.log.info("delete person {}: {}".format(i, dBox))
                    #     continue
                    else:
                        continue
            elif label == "cell":
                index = "cell"
            elif label == "chair":
                index = "chair"
            else:
                pass

            results.append(
                {'resizedName': resized_name, 'objName': self.objName, 'label': label, 'index': index, 'bbox': [xmin, ymin, xmax, ymax]})
        return results
