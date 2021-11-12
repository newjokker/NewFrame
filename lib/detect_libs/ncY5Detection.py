import numpy as np
from .yolov5Detection import YOLOV5Detection


class NcDetection(YOLOV5Detection):
    def __init__(self, args, objName, scriptName):
        super(NcDetection, self).__init__(args, objName, scriptName)

    def merge(self, boxes, classes, scores, thr):
        nc_indices = classes == 0
        nc_boxes = boxes[nc_indices]

        # areas matrices
        areas, inter_areas = self.comptute_areas(nc_boxes, nc_boxes)
        if np.count_nonzero(inter_areas) == len(nc_boxes):
            return self.post_process(boxes, classes, scores)

        else:
            for i in range(len(inter_areas)):
                inter_areas[i, i] = 0
            new_iou = inter_areas / areas
            x_axis = np.where(new_iou > thr)[0]
            rmv = list(set(x_axis))

            keep = np.array([True] * len(classes))
            count = -1
            for i, b in enumerate(nc_indices):
                if b:
                    count += 1
                    if count in rmv:
                        keep[i] = False

            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]
            return self.post_process(boxes, classes, scores)

    def comptute_areas(self, bbox, gt):
        w = bbox[:, 2] - bbox[:, 0] + 1
        h = bbox[:, 3] - bbox[:, 1] + 1
        n_boxes = len(bbox)
        areas = np.tile(w * h, (n_boxes, 1)).transpose()

        lt = np.maximum(bbox[:, None, :2], gt[:, :2])  # left_top (x, y)
        rb = np.minimum(bbox[:, None, 2:], gt[:, 2:])  # right_bottom (x, y)
        wh = np.maximum(rb - lt + 1, 0)  # inter_area (w, h)
        inter_areas = wh[:, :, 0] * wh[:, :, 1]  # shape: (n, m)

        return areas, inter_areas


