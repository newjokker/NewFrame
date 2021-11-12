import numpy as np
from .yolov5Detection import YOLOV5Detection


class JcDetection(YOLOV5Detection):
    def __init__(self, args, objName, scriptName):
        super(JcDetection, self).__init__(args, objName, scriptName)

    def getRoI(self, objects):
        td = [obj for obj in objects if obj[0] == 'td']
        bhm = [obj for obj in objects if obj[0] == 'bhm']

        if len(td) == 0:
            tj = [obj for obj in objects if obj[0] == 'tj']
            if len(bhm) > len(tj):
                points = bhm
            else:
                points = tj

            if len(points) >= 3:
                n = len(objects)
                boxes = [obj[2:-1] for obj in points]
                boxes = np.array(boxes)
                xmin = min(boxes[:, 0])
                ymin = min(boxes[:, 1])
                xmax = max(boxes[:, 2])
                ymax = max(boxes[:, 3])

                td = [['td', n, xmin, ymin, xmax, ymax, 1.0]]

        results = []
        results.extend(td)
        results.extend(bhm)
        return results
