import os, sys
import tensorflow as tf
import numpy as np
import cv2
import configparser
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except,timeStamp
from ..detect_utils.utils import isBoxExist
from ..detect_utils.cryption import salt, decrypt_file
from ..detect_libs.deeplabDetection import DeeplabDetection
import heapq
import math
import struct
from PIL import Image

class jyhDeeplabDetection(DeeplabDetection):
    def __init__(self,args,objName,scriptName):
        super(jyhDeeplabDetection,self).__init__(args,objName,scriptName)

    @try_except()
    def postProcess(self,im,image_name):
        self.log.line('postProcess')
        contourList = self.contours_image(im,image_name,2)
        if len(contourList) == 0:
            return None
        result = self.fitEllipse(contourList)
        if result == {}:
            return None
        self.log.info(result)
        return self.getExPointsOfEllipse(image_name,result)

    @try_except()
    def contours_image(self,image,image_name='default.jpg',contour_num=1):
        contours, hierarchy = self.find_contours_with_erode(image) 
        if len(contours) < contour_num:
            self.log.line('len(contours<2)')
            self.log.info(len(contours))
            return []
        contourList = self.find_max_contous(image,contours,2)
        return contourList
   
    @try_except()
    def fitEllipse(self,contourList):
        array = []
        for contour in contourList:
            #拟合椭圆
            if len(contour) < 5:
                return {}
            ellipse = cv2.fitEllipse(contour)
            array.append((int(ellipse[0][0]),int(ellipse[0][1])))
            self.log.info(ellipse[2])
            if  (ellipse[1][1]/ellipse[1][0]) < 1.4:
                self.log.info("长短轴比：" + str(ellipse[1][1]/ellipse[1][0]))
                return {}
            #cv2.ellipse(image,ellipse,(0,255,0),2)

            #求椭圆的极值点
            x1 = ellipse[0][0] + math.cos(math.pi * (ellipse[2]-90)/180) * ellipse[1][1] / 2
            y1 = ellipse[0][1] + math.sin(math.pi * (ellipse[2]-90)/180) * ellipse[1][1] / 2
            x2 = ellipse[0][0] - math.cos(math.pi * (ellipse[2]-90)/180) * ellipse[1][1] / 2
            y2 = ellipse[0][1] - math.sin(math.pi * (ellipse[2]-90)/180) * ellipse[1][1] / 2
            #cv2.circle(image, (int(x1),int(y1)), 2, (255, 255, 255), 3)
            #cv2.circle(image, (int(x2),int(y2)), 2, (255, 255, 255), 3)
            array.append((int(x2),int(y2)))
            array.append((int(x1),int(y1)))
        return array

    @try_except()
    def getExPointsOfEllipse(self,image_name,array):
        #array[0]大轮廓中心
        #array[1]和array[2]大轮廓两个极值点
        #array[3]小轮廓中心
        #array[4]和array[5]小轮廓两个极值点
        listText = {}
        self.log.line('getExPointsOfEllipse')
        self.log.info(array)
        ang = angle(array[1], array[2], array[4], array[5])
        self.log.info("ang = " + str(ang))
        if ang < 6 or ang > 170:
            m,n = GetIntersectPointofLines(array[1], array[2], array[4], array[5])
            Hstart = tuple(array[1])
            Hend = tuple(array[2])
            Vend = (int((array[4][0] + array[5][0])/2),int((array[4][1] + array[5][1])/2))
            Vstart = (int(m),int(n))
        else:
            Hstart = tuple(array[1])
            Hend = tuple(array[2])
            Vend = tuple(array[3])
            Vstart = (int((array[1][0] + array[2][0])/2),int((array[1][1] + array[2][1])/2))
            ang = angle(Vstart, Hend, Vstart, Vend)
            self.log.info("ang2 = " + str(ang))
            if abs(ang-90) >= 30:
                Hstart = tuple(array[4])
                Hend = tuple(array[5])
                Vend = tuple(array[0])
                Vstart = (int((array[4][0] + array[5][0])/2),int((array[4][1] + array[5][1])/2))

        if get_jyhangle(array[0],array[1], array[2], array[3], array[4], array[5]):
            Hstart = tuple(array[4])
            Hend = tuple(array[5])
            Vend = tuple(array[0])
            Vstart = (int((array[4][0] + array[5][0])/2),int((array[4][1] + array[5][1])/2))

        #cv2.circle(image, Hstart, 2, (255, 255, 255), 3)
        #cv2.circle(image, Hend, 2, (255, 255, 255), 3)
        #cv2.circle(image, Vstart, 2, (255, 255, 255), 3)
        #cv2.circle(image, Vend, 2, (255, 255, 255), 3)

        ang = angle((Vstart[0],Vstart[1]), (Hend[0],Hend[1]), (Vstart[0],Vstart[1]), (Vend[0],Vend[1]))
        self.log.info("angle is:",str(ang))
        if abs(ang-90)>45:
            return {}
        
        if self.debug:
            if 'dot' in self.tmpPaths.keys():
                pass#cv2.imwrite(os.path.join(self.tmpPaths['dot'], image_name + '.jpg'), image)

        coordinate = []
        coordinate.append(Hstart)
        coordinate.append(Hend)
        coordinate.append(Vstart)
        coordinate.append(Vend)
        listText['angle'] = ang
        listText['xVstart'] = Vstart[0]
        listText['yVstart'] = Vstart[1]
        listText['xHend'] = Hend[0]
        listText['yHend'] = Hend[1]
        listText['xVend'] = Vend[0]
        listText['yVend'] = Vend[1]
        return listText
         
def angle(v1, v2, v3, v4):
    dx1 = v2[0] - v1[0]
    dy1 = v2[1] - v1[1]
    dx2 = v4[0] - v3[0]
    dy2 = v4[1] - v3[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # MYLOG.info(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # MYLOG.info(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
    if included_angle > 180:
        included_angle = 360 - included_angle
    return included_angle

def get_jyhangle(v0,v1,v2,v3,v4,v5):
    len1 = (v2[0]-v1[0])**2 + (v2[1]-v1[1])**2
    len2 = (v5[0]-v4[0])**2 + (v5[1]-v4[1])**2

    if v0[1] > v3[1] and len1/len2 < 1.5:
        return True
    else:
        return False

def GetIntersectPointofLines(v1, v2, v3, v4):
    A1,B1,C1=medLine(v3[0],v3[1],v4[0],v4[1])
    A2,B2,C2=GeneralEquation(v1[0],v1[1],v2[0],v2[1])
    m=A1*B2-A2*B1
    x = 1
    y = 1
    if m==0:
        print("无交点")
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

