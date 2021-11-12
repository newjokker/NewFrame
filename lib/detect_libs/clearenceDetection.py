import os
import cv2
import numpy as np
from ..detect_utils.utils import mat_inter, check_record_file_path
from ..detect_utils.tryexcept import try_except
from ..detect_libs.fasterDetection import FasterDetection


class ClearenceDetection(FasterDetection):
    def __init__(self, args, objName, scriptName):
        super(ClearenceDetection, self).__init__(args, objName, scriptName)

        self.lssd_resizedCapPath = self.getTmpPath('lssd_resizedCap')
        self.allDetectedPath = self.getTmpPath('allDetected')

        if self.debug:
            ### 生成（反标）要筛出的标签截图 ###
            self.labeles_checkedOut = [l for l in self.CLASSES if  "other" in l]
            self.labeles_checkedIn = list(set(self.CLASSES) - set(self.labeles_checkedOut))


    @try_except()
    def checkDetectBoxAreas(self, tot_label):
        if self.areaThresh > 0:
            return [obj for obj in tot_label if (obj[4]-obj[2])*(obj[5]-obj[3])>self.areaThresh]
        else:
            return tot_label


    @try_except()
    def postProcess(self, im, kkx_obj):

        resizedName = kkx_obj['kkxName']
        self.log.info("resizedName {} shape:{}".format(resizedName, im.shape))

        ###写字#####
        font_style = cv2.FONT_HERSHEY_SIMPLEX  # 字体
        font_size = 6
        font_cuxi = 6

        detected_boxes = {}  ### key为检测框类 value为坐标box #
        detectedLables = []
        for i, obj in enumerate(self.tot_labels):
                label, i, xmin, ymin, xmax, ymax = obj[:-1]
                box_i = (xmin, ymin, xmax, ymax)
                detected_boxes[label] = box_i
                detectedLables.append(label)
                color = self.color_mathes_label(label)
                ### 画检测目标 ###
                cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 6)
                cv2.putText(im, label, (xmin, ymin - 3), font_style, font_size, color, font_cuxi)

        # 如果检测框重合 detectedLables,置为suspicious####
        label_coincided_flag = self.isDetectedBoxes_Coincided(detected_boxes)  ## True 为重合 False 为不重合
        detected_rs = self.rs_mathes_labels(detectedLables,label_coincided_flag)

        if self.debug:
            cv2.imwrite(os.path.join(self.allDetectedPath, resizedName + '.jpg'), im)

        return detected_rs


    @try_except()
    def rs_mathes_labels(self,labels,coincided_flag):
        detected_output = ""
        if coincided_flag:
            detected_output = "suspicious"
        else:
            if "jx" in labels or "jxd" in labels:
                detected_output= "jx"
            else:
                detected_output = "th"
        return detected_output


    @try_except()
    def color_mathes_label(self,labelCls):
        ###定义标签颜色#####
        color_jx = (114, 128, 250)
        color_th = (127, 255, 0)
        color_others = (238, 238, 0)

        if "jx" in labelCls:
            return color_jx
        elif "th" == labelCls:
            return color_th
        elif "others" in labelCls:
            return color_others

    @try_except()
    def isDetectedBoxes_Coincided(self,detected_boxes):

        Flag = False ## True 为重合 False 为不重合 ###
        ### 先分为两部分 ###
        clearance_boxes = {}  # jx jxd
        normal_boxes = {}  # others others2 th

        for key in detected_boxes.keys():
            if "jx" in key or "jxd" in key:
                clearance_boxes[key] = detected_boxes[key]
            else:
                normal_boxes[key] = detected_boxes[key]

        ### 如果检测到松动，进入判断 ###
        isInNum = 0
        if len(clearance_boxes.keys()) > 0:
            for key in clearance_boxes.keys():
                isInNum += self.isBoxIn2(clearance_boxes[key], normal_boxes)
        if isInNum > 0:
            Flag = True
            #### 将此设置为suspious ####
        return Flag

    @try_except()
    def isBoxIn2(self,newbox, boxes):
        for key in boxes.keys():
            if (self.solve_coincide(boxes[key], newbox)):  # 计算重合面积#  > 0.6
                return 1
        return 0

    @try_except()
    def solve_coincide(self,box1, box2):
        if self.mat_inter(box1, box2) == True:
            x01, y01, x02, y02 = box1
            x11, y11, x12, y12 = box2
            col = min(x02, x12) - max(x01, x11)
            row = min(y02, y12) - max(y01, y11)
            intersection = col * row
            area1 = (x02 - x01) * (y02 - y01)
            area2 = (x12 - x11) * (y12 - y11)
            coincide = intersection / max(area1, area2)
            if coincide > 0.5:
                return True
            else:

                return False
        else:
            return False

    @try_except()
    def mat_inter(self,box1, box2):
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
        ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
        sax = abs(x01 - x02)
        sbx = abs(x11 - x12)
        say = abs(y01 - y02)
        sby = abs(y11 - y12)
        if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
            return True
        else:
            return False