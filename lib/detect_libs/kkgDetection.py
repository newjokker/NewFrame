import os
import cv2
import numpy as np
from ..detect_utils.utils import mat_inter, check_record_file_path
from ..detect_utils.tryexcept import try_except
from ..detect_libs.fasterDetection import FasterDetection


class KkgDetection(FasterDetection):
    def __init__(self, args, objName, scriptName):
        super(KkgDetection, self).__init__(args, objName, scriptName)
        self.testImgPath = self.getTmpPath('testImg')
        self.allDetectedPath = self.getTmpPath('allDetected')
        self.AllDetectRS_picPath = self.getTmpPath('AllDetectRS_pic')

        self.resizedImgPath = os.path.join(self.tmpPath, 'ljc_demo', 'resizedImg')
        self.filterFuncs.extend([self.checkDetectBoxAreas])

        if self.debug:
            ### 生成（反标）要筛出的标签截图 ###
            self.labeles_checkedOut = [l for l in self.CLASSES if "dense" in l or "other" in l]
            self.labeles_checkedIn = list(set(self.CLASSES) - set(self.labeles_checkedOut))


    @try_except()
    def checkDetectBoxAreas(self, tot_label):
        if self.areaThresh > 0:
            return [obj for obj in tot_label if (obj[4]-obj[2])*(obj[5]-obj[3])>self.areaThresh]
        else:
            return tot_label


    @try_except()
    def postProcess(self, im, ljcBox):
        if self.debug:
            ljc_name = ljcBox['resizedName']
            self.log.info("ljcBox {} shape:{}".format(ljc_name, im.shape))

            ###写字#####
            font_style = cv2.FONT_HERSHEY_SIMPLEX  # 字体
            font_size = 1
            font_cuxi = 2

            for i, obj in enumerate(self.tot_labels):
                label, i, xmin, ymin, xmax, ymax = obj[:-1]

                if label in self.labeles_checkedOut:
                    color = (114, 128, 250)  # red
                elif label == "Lm":
                    color = (255, 0, 0)  # blue
                else:
                    color = (0, 255, 0)  # green

                ### 画检测目标 ###
                cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 3)
                cv2.putText(im, label, (xmin, ymin - 3), font_style, font_size, color, font_cuxi)
            cv2.imwrite(os.path.join(self.allDetectedPath, ljc_name + '.jpg'), im)


    @try_except()
    def draw_all_boxes(self, img_name, ljcBoxes, results):
        if self.debug:
            ori_name = img_name + '.jpg' if os.path.exists(os.path.join(self.WORKDIR, 'inputImg', img_name + '.jpg')) else img_name + '.JPG'
            ori_img_path = os.path.join(self.WORKDIR, 'inputImg', ori_name)

            if os.path.exists(ori_img_path):
                resultImg = cv2.imdecode(np.fromfile(ori_img_path, dtype=np.uint8), 1)

                ###写字#####
                font_style = cv2.FONT_HERSHEY_SIMPLEX  # 字体
                font_size = 1
                font_cuxi = 2

                for resizedName, objects in results.items():
                    ljc_result = [obj for obj in ljcBoxes if obj['resizedName'] == resizedName]
                    xmap = ljc_result[0]['xmin']
                    ymap = ljc_result[0]['ymin']

                    objects = objects[1:]

                    for obj in objects:
                        label, i, xmin, ymin, xmax, ymax = obj[:-1]
                        xmin, ymin, xmax, ymax = xmin+xmap, ymin+ymap, xmax+xmap, ymax+ymap

                        if label in self.labeles_checkedOut:
                            cv2.rectangle(resultImg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (114, 128, 250), 3)  # red 114, 128, 250
                            cv2.putText(resultImg, label, (xmin, ymin - 3), font_style, font_size, (114, 128, 250), font_cuxi)
                        elif "Lm" in label:
                            cv2.rectangle(resultImg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)  # blue
                            cv2.putText(resultImg, label, (xmin, ymin - 3), font_style, font_size, (255, 0, 0), font_cuxi)
                        elif "TK" in label:
                            cv2.rectangle(resultImg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 3)  # blue
                            cv2.putText(resultImg, label, (xmin, ymin - 3), font_style, font_size, (255, 255, 0), font_cuxi)
                        else:
                            cv2.rectangle(resultImg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)  # green
                            cv2.putText(resultImg, label, (xmin, ymin - 3), font_style, font_size, (0, 255, 0), font_cuxi)

                # draw_ljcs #
                for ljcObj in ljcBoxes:
                    xmin = int(ljcObj['xmin'])
                    ymin = int(ljcObj['ymin'])
                    xmax = int(ljcObj['xmax'])
                    ymax = int(ljcObj['ymax'])
                    label = ljcObj['label']
                    ljc_resizedName = ljcObj['resizedName']
                    ljc_obj = ljc_resizedName.split('_resized_')[-1].split('_')[0]

                    font_style = cv2.FONT_HERSHEY_SIMPLEX  # 字体
                    font_size = 1
                    font_cuxi = 3

                    cv2.rectangle(resultImg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (238, 238, 141), 3)  # yellow
                    cv2.putText(resultImg, label+'_'+str(ljc_obj), (xmin, ymin - 3), font_style, font_size, (238, 238, 141), font_cuxi)

                cv2.imwrite(os.path.join(self.AllDetectRS_picPath, ori_name), resultImg)


    @try_except()
    def checkOthersGlobal(self, kkxBoxes, ljcBoxes):
        try:
            self.log.info('in checkOthersGlobal')
            otherList = []
            kList = {}

            for ljcBox in ljcBoxes:
                ljcName = ljcBox['resizedName']
                xmap, ymap = ljcBox['xmin'], ljcBox['ymin']
                kList[ljcName] = []

                ## 因为之前有的ljcBox由于过滤后为[]，被删除，所以此处要加判断 ##
                if ljcName in kkxBoxes.keys():
                    for kkxBox in kkxBoxes[ljcName]:
                        kkxBox[2] += xmap
                        kkxBox[3] += ymap
                        kkxBox[4] += xmap
                        kkxBox[5] += ymap
                        if ('other' in kkxBox[0]):
                            otherList.append((float(kkxBox[2]), float(kkxBox[3]), float(kkxBox[4]), float(kkxBox[5])))
                        else:
                            kList[ljcName].append(kkxBox)

            b = {}
            self.log.info('44444444444444444444444444')
            self.log.info(len(otherList))
            self.log.info(otherList)
            self.log.info('55555555555555555555555555')
            self.log.info(kList)
            self.log.info('66666666666666666666666666')
            for ljcName, ks in kList.items():
                b[ljcName] = []
                for k in ks:
                    self.log.info(k)
                    if self.isBoxIn_other(otherList, (float(k[2]), float(k[3]), float(k[4]), float(k[5]))):
                        self.log.info(" k in global_others:", k)
                        continue
                    else:
                        b[ljcName].append(k)
            return b
        except Exception as e:
            self.log.info('GOT ERROR---->')
            self.log.info(e)
            self.log.info(e.__traceback__.tb_frame.f_globals["__file__"])
            self.log.info(e.__traceback__.tb_lineno)


    @try_except()
    def postProcessGlobalFlex(self, imgs, name, detectBoxes, ljcBoxes,loc_definition,extendRate):
        try:
            results = []
            for ljcBox in ljcBoxes:
                ljcName = ljcBox['resizedName']
                ## global：相对原图坐标, partial：相对截图坐标
                position_definition = {'global':(0, 0),'partial':(ljcBox['xmin'], ljcBox['ymin'])}
                xmap, ymap = position_definition[loc_definition]
                xmap_c = ljcBox['xmin'] - xmap
                ymap_c = ljcBox['ymin'] - ymap
                i = 0
                im = imgs[ljcName]
                self.log.info('detectBoxes')
                self.log.info(detectBoxes)

                for detectBox in detectBoxes[ljcName]:
                    self.log.info(detectBox)
                    label = self.getFinalLabel(str(detectBox[0]))
                    resizedName = ljcName + "_kkg_" + str(i)
                    prob = str(detectBox[6])
                    #  - xmap, - ymap 即为相对截图坐标
                    xmin = int(float(str(detectBox[2])) - xmap)
                    ymin = int(float(str(detectBox[3])) - ymap)
                    xmax = int(float(str(detectBox[4])) - xmap)
                    ymax = int(float(str(detectBox[5])) - ymap)
                    ## 扩增截图范围 ,单边 extendRate/2##
                    height, width, _ = im.shape
                    new_xmin = int(-(xmax - xmin) * (extendRate / 2) + xmin)
                    new_xmax = int((xmax - xmin) * (extendRate / 2) + xmax)
                    new_ymin = int(-(ymax - ymin) * (extendRate / 2) + ymin)
                    new_ymax = int((ymax - ymin) * (extendRate / 2) + ymax)

                    resultImg = im[(new_ymin - ymap_c):(new_ymax - ymap_c), (new_xmin - xmap_c):(new_xmax - xmap_c)]

                    self.log.info(resizedName)
                    self.log.info(label, prob)
                    results.append(
                        {'kkxName': resizedName, 'ljcName': ljcName, 'label': label, 'prob': prob, 'index': i,
                         'xmin': new_xmin, 'ymin': new_ymin, 'xmax': new_xmax, 'ymax': new_ymax})
                    cv2.imwrite(os.path.join(self.testImgPath, resizedName + '.jpg'), resultImg)
                    i += 1
            return results

        except Exception as e:
            self.log.info('GOT ERROR---->')
            self.log.info(e)
            self.log.info(e.__traceback__.tb_frame.f_globals["__file__"])
            self.log.info(e.__traceback__.tb_lineno)


    def getFinalLabel(self, originLabel):
        if originLabel == 'KG':
            return 'Xnormal'
        elif originLabel == 'K':
            return 'Xnormal'
        else:
            return originLabel


    @try_except()
    def isBoxIn_other(self, boxList, bbox):
        for box in boxList:
            if (self.solve_coincide_other(box, bbox, 0.8)):
                return True
        return False


    @try_except()
    def solve_coincide_other(self, box1, box2, iouThresh):
        if mat_inter(box1, box2) == True:
            coincide = self.compute_iou_other(box1, box2)
            return coincide > iouThresh
        else:
            return False


    @try_except()
    def compute_iou_other(self, box1, box2):
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / min(area1, area2)
        return coincide



