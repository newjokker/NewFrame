import os
import cv2
#from ..detect_libs.fasterDetection import FasterDetection
from ..detect_utils.tryexcept import try_except
#from ..detect_utils.utils import compute_iou
from ..detect_libs.yolov5Detection import YOLOV5Detection


def compute_iou(pos1,pos2):
    left1,top1,right1,down1 = pos1
    left2,top2,right2,down2 = pos2
    area1 = (right1-left1)*(down1 - top1)
    area2 = (right2-left2)*(down2 - top2)
    area_sum = area1 + area2
    # 计算中间重叠区域的坐标
    left = max(left1,left2)
    right = min(right1,right2)
    top = max(top1,top2)
    bottom = min(down1,down2)
    if left >= right or top >= bottom:
        return 0
    else:
        inter = (right-left)*(bottom-top)
        return inter/min(area1,area2)


class jyhTLDetection(YOLOV5Detection):

    def __init__(self, args, objName,scriptName):
        super(jyhTLDetection, self).__init__(args, objName,scriptName)

    @try_except()
    def postProcess(self,im,image_name,jyzBoxes):
        H,W,_=im.shape
        results={}
        fhjyz = []
        jyh = []

        fhjyz_dete_res=[obj for obj in jyzBoxes if obj[0]=="fhjyz"]
        jyh_dete_res=[obj for obj in jyzBoxes if obj[0] in ["upring","downring","wairing","leftring"]]

        for jyz_dete in fhjyz_dete_res:
            jyh_num = 0
            box_jyz = jyz_dete[2:6]
            for jyh_dete in jyh_dete_res:
                box_jyh = jyh_dete[2:6]
                jyh_label = jyh_dete[0]  # 判断属于什么均压环 upring,downring还是，根据这个结果做位置判断
                iou_value = compute_iou(box_jyz, box_jyh)

                if iou_value > 0.2:
                    if jyh_label == "upring" and box_jyz[1] < box_jyh[3] and box_jyz[1] > box_jyh[1]:
                        jyh_num += 1
                    if jyh_label == "downring" and box_jyz[3] < box_jyh[3] and box_jyz[3] > box_jyh[1]:
                        jyh_num += 1
                    if jyh_label == "leftring":
                        if (box_jyz[0] > box_jyh[0] and box_jyz[0] < box_jyh[2]) or (box_jyz[2] > box_jyh[0] and box_jyz[2] < box_jyh[2]):
                            jyh_num+=1
                    if jyh_label == "wairing":
                        if (box_jyz[1] < box_jyh[3] and box_jyz[1] > box_jyh[1]) or (box_jyz[3] < box_jyh[3] and box_jyz[3] > box_jyh[1]):
                            jyh_num += 1

            if jyz_dete[3] > 20 and jyz_dete[5] < H - 20 and jyz_dete[2] > 20 and jyz_dete[4] < W - 20:
                N_jyh=2
            elif (jyz_dete[3] < 20 and jyz_dete[5] > H - 20) or (jyz_dete[2] < 20 and jyz_dete[4] > W - 20):
                N_jyh=0
            else:
                N_jyh=1
            if max(0,N_jyh-jyh_num) == 0:
                stat="Exist"
            else:
                stat="Miss"
            resizedName=image_name + "_resized_{}".format(jyz_dete[1])

            results[resizedName]= {'label':jyz_dete[0],'index':jyz_dete[1],'xmin':jyz_dete[2],'ymin':jyz_dete[3],'xmax':jyz_dete[4],'ymax':jyz_dete[5],'class':stat}

        return results


