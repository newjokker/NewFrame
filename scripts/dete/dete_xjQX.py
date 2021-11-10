# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import cv2
import xml.etree.ElementTree as ET
import configparser
import math
import time

from lib.JoTools.txkjRes.resTools import ResTools
from lib.JoTools.utils.FileOperationUtil import FileOperationUtil
from lib.JoTools.utils.CsvUtil import CsvUtil
from lib.JoTools.txkjRes.deteRes import DeteRes
from lib.JoTools.txkjRes.deteObj import DeteObj
from lib.JoTools.txkjRes.deteAngleObj import DeteAngleObj
from lib.JoTools.utils.JsonUtil import JsonUtil
#
from JoTools.utils.DecoratorUtil import DecoratorUtil
import copy

cwd = os.path.dirname(os.path.abspath(__file__))

@DecoratorUtil.time_this
def dete_xjQX(model_dict, data):
    try:
        model_xjQX_1 = model_dict["model_xjQX_1"]
        model_xjQX_2 = model_dict["model_xjQX_2"]
        #
        start_time = time.time()
        xjQX_dete_res = DeteRes()
        detectBoxes = model_xjQX_1.detect(data['im'], data['name'])
        # todo 重写第一步的后处理，所有要新加的逻辑都放在第一步的后处理

        results = model_xjQX_1.postProcess2(data['im'], data['name'], detectBoxes)

        stop_model_1_time = time.time()

        for xjBox in results:
            resizedName = xjBox['resizedName']
            resizedImg = data['im'][xjBox['ymin']:xjBox['ymax'], xjBox['xmin']:xjBox['xmax']]
            segImage = model_xjQX_2.detect(resizedImg, resizedName)
            result = model_xjQX_2.postProcess(segImage, resizedName, xjBox)
            # add obj
            if "position" in result:
                x1, y1, w, h = result["position"]
                x2, y2 = x1 + w, y1 + h
                tag = result["class"]
                xjQX_dete_res.add_obj(x1, y1, x2, y2, tag, conf=-1, assign_id=-1, describe='')
        end_time = time.time()
        print("* step 1 : {0}".format(stop_model_1_time - start_time))
        print("* step 2 : {0}".format(end_time - stop_model_1_time))
        # torch.cuda.empty_cache()
        return xjQX_dete_res
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)

def normal_or_fail(ang, theta):
    """判断是否倾斜，根据均压环横轴与绝缘子纵轴之间的角度与指定阈值之间的关系"""
    if abs(ang - 90) > theta:
        return 'fail'
    else:
        return 'Hnormal'

def Judge_Hnormal_fail(angle1, angle2, assign_theta=10):
    """根据两个角度判断是否倾斜"""
    if angle2 is None:
        return normal_or_fail(angle1, assign_theta), 0, angle1
    elif (abs(angle2 - 90) > assign_theta) and (abs(angle2 - 90) > abs(angle1 - 90)):
        return normal_or_fail(angle1, assign_theta), 1, angle2
    else:
        return normal_or_fail(angle2, assign_theta), 0, angle1

# @DecoratorUtil.time_this
def angle_r2cnn(x_add, y_add, each_dete_obj, result, des):
    """angle r2cnn """
    x_start, y_start, x_hend, y_hend, x_vend, y_vend = result['xVstart'], result['yVstart'], result['xHend'], result['yHend'], result['xVend'], result['yVend']
    X_start, Y_start, X_hend, Y_hend, X_vend, Y_vend = x_start + x_add, y_start + y_add, x_hend + x_add, y_hend + y_add, x_vend + x_add, y_vend + y_add
    x1, y1, x2, y2 = eval(des)
    X1, Y1, X2, Y2 = x1 + x_add, y1 + y_add, x2 + x_add, y2 + y_add

    jyh_info, jyz_info = {}, {}
    jyh_info['xstart'] = X_start
    jyh_info['ystart'] = Y_start
    jyh_info['xHend'] = X_hend
    jyh_info['yHend'] = Y_hend
    jyh_info['xVend'] = X_vend
    jyh_info['yVend'] = Y_vend
    jyh_info['angle'] = result['angle']
    jyh_info['label'] = each_dete_obj.tag

    jyz_info['x1'] = X1
    jyz_info['x2'] = X2
    jyz_info['y1'] = Y1
    jyz_info['y2'] = Y2
    # print("jyh_info:{}".format(jyh_info))
    # print("jyz_info:{}".format(jyz_info))
    jyz_angle, new_jyh_info = caculateJyzAngleAndIntersectionWithJyh(jyh_info, jyz_info)
    return jyz_angle

# @DecoratorUtil.time_this
def caculateJyzAngleAndIntersectionWithJyh(jyh_info,jyz_info):
    #映射回原图尺寸
    #for jyz in jyzResult:
    x1 = jyz_info['x1']
    y1 = jyz_info['y1']
    x2 = jyz_info['x2']
    y2 = jyz_info['y2']
    #中点
    x3=0.5*(x1+x2)
    y3=0.5*(y1+y2)

    #寻找交点
    intersectionX,intersectionY = cross_point(x1,y1,x2,y2, jyh_info['xstart'],jyh_info['ystart'], jyh_info['xHend'],jyh_info['yHend'])
    jyz_angle = angle((intersectionX,intersectionY), (jyh_info['xHend'],jyh_info['yHend']), (intersectionX,intersectionY), (x3,y3))
    prob = angle_judge(jyz_angle)
    new_angle_obj = jyh_info.copy()
    new_angle_obj['angle'] = jyz_angle
    new_angle_obj['prob'] = prob
    return jyz_angle,new_angle_obj

# @DecoratorUtil.time_this
def cross_point(x1, y1, x2, y2, x3, y3, x4, y4):
    point_is_exist = False
    x = y = 0
   # x1,y1,x2,y2 = line1
   # x3,y3,x4,y4 = line2

    if (x2 - x1) == 0:
        k1 = None
        b1 = 0                                  # fixme 之前没有这一行，我随便给初始化了一个值
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)        # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0           # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            point_is_exist = True
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        point_is_exist = True
    return [x,y]

def angle(v1, v2, v3, v4):
    dx1 = v2[0] - v1[0]
    dy1 = v2[1] - v1[1]
    dx2 = v4[0] - v3[0]
    dy2 = v4[1] - v3[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # MYLOG.info(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # MYLOG.info(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
    if included_angle > 180:
        included_angle = 360 - included_angle
    return included_angle

def angle_judge(angle):
    if abs(angle - 90) > 9:
        return 'fail'
    else:
        return 'Hnormal'

def get_lines_to_draw(line_jyh, line_jyz_1, line_jyz_2, line_index, x_add, y_add, use_angle):
    """得到要画的两条线段"""
    # 用四边形两对对角坐标相加相等的原理，求出另外一个点与 xstart,ystart 组成一个新的直线

    jyh_x1, jyh_y1, jyh_x2, jyh_y2 = line_jyh
    jyz_1_x1, jyz_1_y1, jyz_1_x2, jyz_1_y2 = line_jyz_1
    jyz_2_x1, jyz_2_y1, jyz_2_x2, jyz_2_y2 = line_jyz_2

    if line_index == 0:
        return [[jyh_x1 + x_add, jyh_y1 + y_add], [jyh_x2 + x_add, jyh_y2 + y_add],
                [jyz_1_x1 + x_add, jyz_1_y1 + y_add], [jyz_1_x2 + x_add, jyz_1_y2 + y_add], use_angle]
    else:
        # 计算对称点
        jyh_x1, jyh_y1, jyh_x2, jyh_y2 = line_jyh
        jyz_new_x = jyz_2_x2 + jyh_x1 - jyz_2_x1
        jyz_new_y = jyz_2_y2 + jyh_y1 - jyz_2_y1
        # return
        return [[jyh_x1 + x_add, jyh_y1 + y_add], [jyh_x2 + x_add, jyh_y2 + y_add],
                [jyz_1_x1 + x_add, jyz_1_y1 + y_add], [jyz_new_x + x_add, jyz_new_y + y_add], use_angle]

def screen(y, img):
    # screen brightness
    _, _, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    vmedian = np.median(v)
    if vmedian < 35:
        y = '0'
    # screen obscure
    blurry = cv2.Laplacian(img, cv2.CV_64F).var()
    if blurry < 200:
        y = '0'
    return y


