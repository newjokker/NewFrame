import os,sys
import cv2
import math
import configparser
import tensorflow as tf
import numpy as np
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except,timeStamp
from ..detect_utils.utils import isBoxExist
from ..detect_utils.cryption import salt, decrypt_file
from ..detect_libs.deeplabDetection import DeeplabDetection
from scipy import spatial
from sklearn.cluster import DBSCAN

class xjDeeplabDetection(DeeplabDetection):
    def __init__(self,args,objName,scriptName):
        super(xjDeeplabDetection,self).__init__(args,objName,scriptName)

    @try_except()
    def postProcess(self,segImage,resizedName,xjBox):

        segImgPath = self.getTmpPath('segImg')
        cv2.imwrite(os.path.join(segImgPath, resizedName + '.jpg'),segImage)
        seg_map = cv2.imread(os.path.join(segImgPath, resizedName + '.jpg'),-1)
        result = self.do_contours(resizedName,seg_map)
        self.log.info('contours_image_result:')
        self.log.info(result)

        if result != {}:
            result['xmin'] = int(xjBox['xmin'])
            result['xmax'] = int(xjBox['xmax'])
            result['ymin'] = int(xjBox['ymin'])
            result['ymax'] = int(xjBox['ymax'])
            alarm = self.compute_angle(result)
            if alarm != {}:
                return alarm
        else:
            return {}

    @try_except()
    def do_contours(self,image_name,original_image):
        #处理轮廓的总函数
        #prepare pic
        kernel = np.ones((5, 5), np.uint8)
        # 开运算，小点去掉
        image = cv2.morphologyEx(original_image, cv2.MORPH_OPEN, kernel)

        #find contours
        contours, hierarchy = self.find_contours(image)
        self.log.info('contours '+str(len(contours)))
       
        #find 两个线段（一横一竖）黏连在一起的轮廓
        adhesion_contour = self.find_adhesion_contour(contours, original_image)
        if np.any(adhesion_contour != None):
            self.log.info('have coincident contour')
            contours = [adhesion_contour]
        else:
            self.log.info('have no coincident contour')

        #judge_contours
        is_replace = False
        #黏连轮廓的个数一般就是1个
        if len(contours) <= 1:
            self.log.info('start judge contours')
            new_contours = self.judge_contours(image, contours, original_image)
            if len(new_contours) == 0:
                return {}
            elif len(new_contours) == 1:
                is_replace = True
            else:
                contours = new_contours

        if is_replace:
            #只找到一个横轴，竖轴用垂线代替
            Hend, Vend, start = self.replace_xianjia_by_hor(contours)
        else:
            self.log.info('choose_contour')
            #从多个contour中找出两个
            contour_dict = self.choose_contour(contours)
            self.log.info(contour_dict)
            if len(contour_dict.keys()) < 2:
                return {}
            Hend, Vend, start = self.get_points(contours, contour_dict)
            #Hend = self.expend_xianjia(Hend, Vend, Vstart)
        self.log.info('start:',start) 
        self.log.info('Hend:',Hend)
        self.log.info('Vend:',Vend)
        ang = angle(Vend, start, Hend)
        self.log.info('angle:',ang)
        if ang < 0 or ang > 180:
            return {}

        listText = {}
        listText['resizedName'] = image_name
        listText['xstart'] = int(start[0])
        listText['ystart'] = int(start[1])
        listText['xHend'] = int(Hend[0])
        listText['yHend'] = int(Hend[1])
        listText['xVend'] = int(Vend[0])
        listText['yVend'] = int(Vend[1])

        return listText
   
    @try_except() 
    def judge_contours(self, image, contours, original_image):
        self.log.info('judge_contours')
        # judge contour
        if len(contours) < 1:
            #空的直接返回，说明没有找到contour
            return []

        if len(contours) == 1:
            cnt = contours[0]
            w,h = self.compute_contour_w_h(cnt)
            self.log.info('w,h '+str(w)+" "+str(h))
            #一个框如果长宽比还比较大说明这个框不是两个框黏连
            if w / h > 5 or w / h  < 0.2 :
                self.log.info('just detect one contour')
                return contours
            #这里应该是在做二值化
            height, width = image.shape
            for i in range(1, height - 1):
                for j in range(width):
                    if image[i,j] !=0 and image[i-1,j] > 180 and image[i+1,j] < 180:
                        image[i,j] = 0

            #腐蚀一次再寻找contour
            new_contours, hierarchy = self.find_contours_with_erode(image)
            #再找一次黏连contour
            coincident_contour = self.find_adhesion_contour(new_contours, original_image)
            if np.any(coincident_contour != None):
                #不再做处理了
                self.log.info('have coincident contour in judge contour')
                return {}

            if len(new_contours) < 2:
                #还是没有分离开，放弃处理
                return []
            else:
                return new_contours

    @try_except()
    def getLongAxis(self,boxinfo):
        l1 = self.get_len_between_p1_p2(boxinfo['top'],boxinfo['right'])
        l2 = self.get_len_between_p1_p2(boxinfo['left'],boxinfo['top'])
        if l1 > l2:
            return (self.get_middle_point(boxinfo['top'],boxinfo['left']),self.get_middle_point(boxinfo['right'],boxinfo['bottom'])) 
        else:
            return (self.get_middle_point(boxinfo['left'],boxinfo['bottom']),self.get_middle_point(boxinfo['top'],boxinfo['right']))


    @try_except()
    def get_points(self, contours, contour_dict):
        segment = []
        segment_backup = []
        for index,item in contour_dict.items():
            contour = contours[index]
            i,j = self.get_two_point_with_longest_distance(contour)
            self.log.info('i:',i)
            self.log.info('j:',j)
            long_axis = self.getLongAxis(item)
            #获取轮廓距离最长的两个点，和外接矩形长边的中轴平行线的两个点，作为备选
            segment_backup.append((tuple(i),tuple(j)))
            segment.append(long_axis)
        #print('##############')
        #print(segment)
        #print(segment_backup)
        #print('##############')
        #根据上面两种备选线段，判断两组挂板和线夹的组合
        guaban,xianjia = self.classify_guaban_xianjia(segment)
        guaban_backup,xianjia_backup = self.classify_guaban_xianjia(segment_backup)
        #分别对挂板线夹做校验，是否符合逻辑 
        if self.check_guaban_xianjia(guaban_backup,xianjia_backup):
            guaban_final,xianjia_final = guaban_backup,xianjia_backup
        elif self.check_guaban_xianjia(guaban,xianjia):
            guaban_final,xianjia_final = guaban,xianjia
        else:
            guaban_final,xianjia_final = guaban,xianjia
        #计算挂板（延长线）和线夹（延长线）的交点
        m,n = GetIntersectPointofLines(xianjia_final[0], xianjia_final[1],guaban_final[0], guaban_final[1])
        mb,nb = GetIntersectPointofLines(xianjia_backup[0], xianjia_backup[1],guaban_backup[0], guaban_backup[1])
        self.log.info('m,n,mb,nb',m,n,mb,nb)
        #为了画图好看，选择的HEND点应该是远端的那个点
        if (m < 0 or m > max(xianjia[0][0],xianjia[1][0])*10) and (mb > 0 and mb < max(xianjia[0][0],xianjia[1][0])*10):
            m = mb
            n = nb
        if (n < 0 or n > max(guaban[0][1],guaban[1][1])*10) and (nb > 0 and nb < max(guaban[0][1],guaban[1][1])*10):
            m = mb 
            n = nb 
        
        if m >= max(xianjia[0][0],xianjia[1][0]):
            if xianjia[0][0] > xianjia[1][0]:
                Hend = tuple(xianjia[1])
            else:
                Hend = tuple(xianjia[0])
        elif m <= min(xianjia[0][0],xianjia[1][0]):
            if xianjia[0][0] > xianjia[1][0]:
                Hend = tuple(xianjia[0])
            else:
                Hend = tuple(xianjia[1])
        else:
            if abs(m-xianjia[0][0])>abs(m-xianjia[1][0]):
                Hend = tuple(xianjia[0])
            else:
                Hend = tuple(xianjia[1])

        if guaban[0][1]>guaban[1][1]: 
            Vend = tuple(guaban[1])
            start = (int(m),int(n))
        else:
            Vend = tuple(guaban[0])
            start = (int(m),int(n))
        self.log.info('m:',m)
        self.log.info('n:',n)
        self.log.info('start:',start)
        self.log.info('Vend:',Vend)
        self.log.info('Hend:',Hend)
        return Hend, Vend, start


    @try_except()
    def check_guaban_xianjia(self,guaban,xianjia):
        self.log.info('check,',guaban,xianjia)
        if guaban[0][0] < guaban[1][0]:
            gp_xmin = guaban[0]
            gp_xmax = guaban[1]
        else:
            gp_xmin = guaban[1]
            gp_xmax = guaban[0]
        if xianjia[0][0] < xianjia[1][0]:
            xp_xmin = xianjia[0]
            xp_xmax = xianjia[1]
        else:
            xp_xmin = xianjia[1]
            xp_xmax = xianjia[0]
        """
           \
            \

           -------
        """
        if gp_xmin[1] < gp_xmax[1]:
            if xp_xmax[0] > gp_xmax[0]:
                self.log.info('111')
                return True
            else:
                return False
        elif gp_xmin[1] > gp_xmax[1]:
            if xp_xmin[0] < gp_xmin[0]:
                self.log.info('222')
                return True
            else:
                return False
        else:
            return False
            
            

    @try_except()
    def classify_guaban_xianjia(self,segment):
        if (segment[0][0][1]+segment[0][1][1])/2 > (segment[1][0][1]+segment[1][1][1])/2:
            guaban = segment[1]
            xianjia = segment[0]
        else:
            guaban = segment[0]
            xianjia = segment[1]
        self.log.info('guaban is:',guaban)
        self.log.info('xianjia is:',xianjia)
        return guaban,xianjia
 
    
    @try_except()
    def get_middle_point(self,p1,p2):
        return ((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)

    @try_except()
    def expend_xianjia(self, Hend, Vend, Vstart):
        lenth = math.sqrt((Vstart[0]-Hend[0])**2 + (Vstart[1]-Hend[1])**2)
        height = math.sqrt((Vstart[0]-Vend[0])**2 + (Vstart[1]-Vend[1])**2)
        self.log.info('lenth '+str(lenth))
        self.log.info('height '+str(height))

        Hend = list(Hend)
        if lenth == 0:
            Hend[0] = Hend[0] + int(0.5 * height)
            Hend[1] = Hend[1]

        if lenth < int(0.2 * height) and lenth > 0:
            cos_theta = (Vstart[1]-Hend[1]) / lenth
            sin_theta = (Vstart[0]-Hend[0]) / lenth
            self.log.info('cos_theta '+str(cos_theta))
            self.log.info('sin_theta '+str(sin_theta))
            k = 0.5 * height
            Hend[0] = int(Hend[0] - k*sin_theta)
            Hend[1] = int(Hend[1] - k*cos_theta)

        return tuple(Hend)

    @try_except()
    def choose_contour(self, contours):
        """
        从n个框中间选择两个最大的
        """
        contour_dict = {}
        for i in range(len(contours)):
            cnt = contours[i]
            self.log.info('#'*20)
            w_h_info = self.compute_contour_boxinfo(cnt)
            #简化信息，只保留每个contour的关键信息:四个点
            contour_dict[i] = w_h_info

        max_two = self.max_contour_items(contour_dict,2)
        #print('contour_dict:',max_two)
        return max_two

    @try_except()
    def replace_xianjia_by_hor(self, contours):
        cnt = contours[0]
        Vend = tuple(cnt[:, 0][cnt[:, :, 1].argmin()])
        Vstart = tuple(cnt[:, 0][cnt[:, :, 1].argmax()])
        height = math.sqrt((Vstart[0]-Vend[0])**2 + (Vstart[1]-Vend[1])**2)
        Hend = (Vstart[0] + int(0.5 * height), Vstart[1])
        Hend = tuple(Hend)
        return Hend, Vend, Vstart

    #找出轮廓大但是其实内部mask小的contours
    @try_except()
    def find_adhesion_contour(self, contours, original_image):
        self.log.info('original_image shape '+str(original_image.shape))
        max_contour = None
        max_area = -100
        for cnt in contours:
            area = cv2.contourArea(cnt)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.insert(box, 4, [box[0][0], box[0][1]], 0)
            box = np.int0(box)

            xmin, ymin = np.min(box[0:4, :], 0)
            xmax, ymax = np.max(box[0:4, :], 0)
            self.log.info('point is:' + str([xmin, ymin, xmax, ymax]))

            box = box.tolist()
            self.log.info('box is:'+str(box))
            non_zero, total_pixels = self.static_nonzero_pixes(box, original_image, xmin, ymin, xmax, ymax)
            self.log.info('non_zero'+str(non_zero))
            self.log.info('total_pixels'+str(total_pixels + 1))
            ratio = float(non_zero/(total_pixels + 1))
            #只有满足轮廓像素数量比上外接矩形面积小于阈值0.3的情况才认为是黏连轮廓
            if ratio < 0.3 and area > max_area:
                max_area = area
                max_contour = cnt
        return max_contour

    @try_except()
    def static_nonzero_pixes(self, box, original_image, xmin, ymin, xmax, ymax):
        self.log.line('static_nonzero_pixes')
        self.log.info(type(original_image))
        self.log.info(original_image.shape)
        h, w= original_image.shape
        non_zero, total_pixels = 0, 0

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        for i in range(ymin, ymax):
            for j in range(xmin, xmax):
                ret = self.ispointinPoly([j, i], box)
                if ret:
                    total_pixels += 1
                    value = original_image[i, j]
                    if value > 100:
                        non_zero += 1
        return non_zero, total_pixels

    @try_except()
    def ispointinPoly(self, point, box):
            N = 0
            for i in range(len(box)-1):
                start = box[i]
                end = box[i+1]
                if start[1] == end[1]:
                    continue
                if point[1] < min(start[1],end[1]):
                    continue
                if point[1] >= max(start[1], end[1]):
                    continue
                x = (point[1] - start[1]) * (end[0] - start[0]) / (end[1] - start[1]) +start[0]
                if x > point[0]:
                    N += 1
            if N % 2 == 0:
                return False
            return True


    @try_except()
    def get_len_between_p1_p2(self,p1,p2):
        length = ((p1[0] - p2[0])**2 + (p1[1]-p2[1])**2)**0.5
        return length 
   
    @try_except()
    def compute_contour_w_h(self,cnt):
        self.log.info('compute_contour_w_h')
        boxinfo = self.compute_contour_boxinfo(cnt)
        l1 = self.get_len_between_p1_p2(boxinfo['top'],boxinfo['right'])
        l2 = self.get_len_between_p1_p2(boxinfo['bottom'],boxinfo['right']) 
        return l1,l2
        
 
    @try_except()
    def compute_contour_boxinfo(self, cnt):
        # 最小外接矩形框，有方向角
        self.log.info('in compute_contour_boxinfo')
        self.log.info(type(cnt))
        self.log.info(cnt)
        rect = cv2.minAreaRect(cnt)
        self.log.info('rect:',rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #获取最左边的点，最右边的点，最上面的点，最下面的点的坐标
        left_point_x = np.min(box[:, 0])
        right_point_x = np.max(box[:, 0])
        top_point_y = np.min(box[:, 1])
        bottom_point_y = np.max(box[:, 1])
        
        left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
        right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
        top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
        bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
        self.log.info("[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y], [right_point_x, right_point_y]")
        self.log.info([top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y], [right_point_x, right_point_y])
        return {'top':(top_point_x,top_point_y),'right':(right_point_x,right_point_y),'bottom':(bottom_point_x,bottom_point_y),'left':(left_point_x,left_point_y)}



    @try_except()
    def max_contour_item(self,contour_dict):
        self.log.info('max_contour_item')
        max_value = 0
        item = {}
        for key in contour_dict.keys():
            boxinfo = contour_dict[key]
            l1 = self.get_len_between_p1_p2(boxinfo['top'],boxinfo['right'])
            l2 = self.get_len_between_p1_p2(boxinfo['bottom'],boxinfo['right'])
            value = l1*l2
            if value > max_value:
                max_value = value
                inx = key
                item = {key:contour_dict[key]}
        return item


    @try_except()
    def max_contour_items(self, contour_dict,num):
        if len(contour_dict.keys()) == 2:
            return contour_dict
        result_contours = {}
        for i in range(num):
            max_contour = self.max_contour_item(contour_dict)
            contour_dict.pop(list(max_contour.keys())[0])
            result_contours.update(max_contour)
        return result_contours

    @try_except()
    def compute_k(self, xstart, ystart, xVend, yVend):
        if (yVend - ystart) == 0:
            k_angle = 0
        else:
            k = (xVend - xstart) / (yVend - ystart)
            k = abs(k)
            ang = math.atan(k)
            k_angle = ang / math.pi * 180
        return k_angle
   
    @try_except()
    def get_two_point_with_longest_distance(self,pts):
        # two points which are fruthest apart will occur as vertices of the convex hull
        pts = np.squeeze(pts)
        candidates = pts[spatial.ConvexHull(pts).vertices]
        # get distances between each pair of candidate points
        dist_mat = spatial.distance_matrix(candidates, candidates)
        # get indices of candidates that are furthest apart
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        return candidates[i],candidates[j]
    
    @try_except()
    def compute_angle(self,box):
        xmin = int(box['xmin'])
        ymin = int(box['ymin'])
        xmax = int(box['xmax'])
        ymax = int(box['ymax'])

        xstart = int(box['xstart']) + int(xmin)
        ystart = int(box['ystart']) + int(ymin)
        xHend = int(box['xHend']) + int(xmin)
        yHend = int(box['yHend']) + int(ymin)
        xVend = int(box['xVend']) + int(xmin)
        yVend = int(box['yVend']) + int(ymin)
        #print('xstart,ystart:',xstart,ystart)
        #print('xVend,yVend:',xVend,yVend)
        #print('xHend,yHend:',xHend,yHend)
        ang = angle((xVend,yVend),(xstart,ystart), (xHend,yHend))
        self.log.info('angle',ang)
        if abs(ang - 90) >= 85:
            #极端异常，大概率检测错误
            return {}
        elif abs(ang - 90) >= 15:
            prob = 'XJfail'
        else:
            prob = 'XJnormal'
        alarm = {}
        self.log.info('class:',prob)
        k_angle = self.compute_k(xstart, ystart, xVend, yVend)
        self.log.info('k_angle:',k_angle)
        if k_angle > 5 and prob == 'XJfail':
            alarm['modelName'] = 'xjQX'
            alarm['class'] = prob
            alarm['posibility'] = 'null'
            alarm['position'] = [xmin, ymin, xmax - xmin, ymax - ymin]
            alarm['issueCode'] = 'null'
            alarm['angle'] = ang
            alarm['others'] = [xstart, ystart, xHend, yHend, xVend, yVend]
        else:
            alarm['modelName'] = 'xjQX'
            alarm['class'] = 'XJnormal'
            alarm['posibility'] = 'null'
            alarm['position'] = [xmin, ymin, xmax - xmin, ymax - ymin]
            alarm['issueCode'] = 'null'
            alarm['angle'] = ang
            alarm['others'] = [xstart, ystart, xHend, yHend, xVend, yVend]

        return alarm


def angle(v1, v2, v3):
    a = math.sqrt((v2[0] - v3[0])**2 + (v2[1] - v3[1])**2)
    b = math.sqrt((v1[0] - v3[0])**2 + (v1[1] - v3[1])**2)
    c = math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
    #cos_a = (b*b + c*c - a*a)/(2*b*c)
    #angle = math.acos(cos_a)
    #angle = int(angle * 180/math.pi)
    B=int(math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c))))
    return B


def GetIntersectPointofLines(v1, v2, v3, v4):
    A1,B1,C1=GeneralEquation(v3[0],v3[1],v4[0],v4[1])
    A2,B2,C2=GeneralEquation(v1[0],v1[1],v2[0],v2[1])
    m=A1*B2-A2*B1
    x = 1
    y = 1
    if m==0:
        pass
        #print("无交点")
        #MYLOG.info("无交点")
    else:
        x=(C2*B1-C1*B2)/m
        y=(C1*A2-C2*A1)/m
    return x,y

def medLine(x1,y1,x2,y2):
    A = 2*(x2-x1)
    B = 2*(y2-y1)
    C = x1**2-x2**2+y1**2-y2**2
    return A,B,C


def GeneralEquation(x1,y1,x2,y2):
    # 一般式 Ax+By+C=0
    A=y2-y1
    B=x1-x2
    C=x2*y1-x1*y2
    return A,B,C
