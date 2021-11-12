import os, sys
import cv2
import numpy as np
from PIL import Image
import heapq
import math
import struct
"""
cv方面的工具类
"""

class SegCvUtils(object):
    def __init__(self):
        pass

    def find_max_contous(self, image, contours=[], contour_num=2,
                         min_point_num = 0, max_point_num = 0):
        if len(contours) == 0:
            contours, _ = self.find_contours_with_erode(image, contour_num)

        if len(contours) < contour_num:
            return []
        nums = []
        for i in range(len(contours)):
            nums.append(len(contours[i]))

        # 找到最大的contour_num个轮廓
        #map(nums.index, heapq.nlargest(contour_num, nums))
        contourList = []
        max_len = 0
        min_len = 0

        for i in range(contour_num):
            index = nums.index(max(nums))
            contourList.append(contours[index])
            nums[index] = 0
            min_len = min(len(contours[index]),min_len)
            max_len = max(len(contours[index]),max_len)
             
        if max_point_num >  min_point_num:
            if min_len < min_point_num or max_len < max_point_num:
                return []
        
        return contourList

    def find_contours_with_erode(self,image):
        image = cv2.erode(image, None, iterations = 3)
        contours, hierarchy = self.find_contours(image)
        return contours, hierarchy
 
    def find_contours(self,image):
        dst = cv2.GaussianBlur(image, (3, 3), 0)
        if len(image.shape) == 3: 
            gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        else:
            gray = dst
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        new_contours, hierarchy = self._find_contours(binary)
        return new_contours, hierarchy

    def _find_contours(self, image, 
                     contour_retrieval_mode = cv2.RETR_EXTERNAL,
                     contour_approximation_method = cv2.CHAIN_APPROX_NONE,
                     draw_contour = True):
        """
        计算图中的轮廓，并返回
        :param image:image
        :param contour_retrieval_mode:轮廓检索模式
        :param contour_num:轮廓近似方法
        :return: contours, result_image
        """
        
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
        return contours,hierarchy

    def find_red_contour(self,img):
        b = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
        g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
        r = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
        b[:, :] = -0.5 * img[:, :, 0]
        g[:, :] = -0.5 * img[:, :, 1]
        r[:, :] = img[:, :, 2]
        image = cv2.merge([b, g, r])  # r通道减去0.5*b&g通道,消除白色的影响,按红色提取目标
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)   # 经验阈值190
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找红色目标的轮廓
        b = []  # 保留其最小外接矩形面积大于500的轮廓;经验阈值500
        for counter in contours:
            s = self.contourArea(counter)
            if s > 500:
                b.append(counter)
        #cv2.drawContours(img, b, -1, (0, 0, 255), 3)  # 画保留的轮廓
        if len(b) > 0:
            rect = cv2.minAreaRect(b[0])
            return rect
        else:
            return []
                                            
    def contourArea(self,cnt):
        # 返回传入轮廓最小外接矩形的面积
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return cv2.contourArea(box)




class CvUtils(SegCvUtils):
    pass
 
