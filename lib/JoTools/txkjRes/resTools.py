# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ..txkjRes.deteObj import DeteObj
from ..txkjRes.deteAngleObj import DeteAngleObj


# todo cal_iou 函数需要进行合并，全部改为新的样式


class ResTools(object):
    """Res 需要的函数"""

    @staticmethod
    def merge_range_list(range_list):
        """正框进行区域合并得到大的区域"""
        x_min_list, y_min_list, x_max_list, y_max_list = [], [], [], []
        for each_range in range_list:
            x_min_list.append(each_range[0])
            y_min_list.append(each_range[1])
            x_max_list.append(each_range[2])
            y_max_list.append(each_range[3])
        return (min(x_min_list), min(y_min_list), max(x_max_list), max(y_max_list))

    @staticmethod
    def region_augment(region_rect, img_size, augment_parameter=None):
        """上下左右指定扩增长宽的比例, augment_parameter, 左右上下"""
        if augment_parameter is None:
            augment_parameter = [0.6, 0.6, 0.1, 0.1]

        widht, height = img_size
        x_min, y_min, x_max, y_max = region_rect
        region_width = int(x_max - x_min)
        region_height = int(y_max - y_min)
        #
        new_x_min = x_min - int(region_width * augment_parameter[0])
        new_x_max = x_max + int(region_width * augment_parameter[1])
        new_y_min = y_min - int(region_height * augment_parameter[2])
        new_y_max = y_max + int(region_height * augment_parameter[3])
        #
        new_x_min = max(0, new_x_min)
        new_y_min = max(0, new_y_min)
        new_x_max = min(widht-1, new_x_max)
        new_y_max = min(height-1, new_y_max)

        return (new_x_min, new_y_min, new_x_max, new_y_max)

    @staticmethod
    def cal_iou(dete_obj_1, dete_obj_2, ignore_tag=False):
        """计算两个检测结果相交程度, ignore_tag 为 True 那么不同标签也计算 iou"""

        if dete_obj_1.tag != dete_obj_2.tag and ignore_tag is False:
            return 0.0

        # 计算两个多边形之间的 iou
        poly_points_list_1 = dete_obj_1.get_points()
        poly_points_list_2 = dete_obj_2.get_points()
        iou = ResTools.polygon_iou(poly_points_list_1, poly_points_list_2)
        return iou

    @staticmethod
    def cal_iou_1(dete_obj_1, dete_obj_2, ignore_tag=True):
        """计算两个矩形框的相交面积，占其中一个矩形框面积的比例 ， """
        if dete_obj_1.tag != dete_obj_2.tag and ignore_tag is False:
            return 0

        # 计算两个多边形之间的 iou
        poly_points_list_1 = dete_obj_1.get_points()
        poly_points_list_2 = dete_obj_2.get_points()
        cover_index = ResTools.polygon_iou_1(poly_points_list_1, poly_points_list_2)
        return cover_index

    @staticmethod
    def polygon_iou(poly_points_list_1, poly_points_list_2):
        """计算任意两个凸多边形之间的 IOU"""
        #
        poly1 = Polygon(poly_points_list_1).convex_hull  # 凸多边形
        poly2 = Polygon(poly_points_list_2).convex_hull  # 凸多边形
        poly3 = poly1.intersection(poly2)
        #
        area_1 = poly1.area
        area_2 = poly2.area
        area_3 = poly3.area
        #
        iou = area_3/(area_1 + area_2 - area_3)
        return iou

    @staticmethod
    def polygon_iou_1(poly_points_list_1, poly_mask_points_list_2):
        """计算一个多边形被另外一个多边形覆盖的比例，覆盖比"""
        poly1 = Polygon(poly_points_list_1).convex_hull  # 凸多边形
        poly2 = Polygon(poly_mask_points_list_2).convex_hull  # 凸多边形
        poly3 = poly1.intersection(poly2)
        #
        area_1 = poly1.area
        area_3 = poly3.area
        #
        cover_index = area_3/area_1
        return cover_index

    @staticmethod
    def crop_angle_rect(img, rect):
        """输入的是弧度，需要转为角度"""
        # get the parameter of the small rectangle
        center, size, angle = rect[0], rect[1], rect[2]
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # get row and col num in img
        # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        height, width = img.shape[0], img.shape[1]
        # calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, (180*angle)/3.14, 1)
        # rotate the original image
        img_rot = cv2.warpAffine(img, M, (width, height))
        # now rotated rectangle becomes vertical and we crop it
        img_crop = cv2.getRectSubPix(img_rot, size, center)
        return img_crop

    @staticmethod
    def point_in_poly(point, ploy_points_list):
        """点是否在多边形中"""
        p1 = Point(point[0], point[1])
        poly1 = Polygon(ploy_points_list).convex_hull
        #
        if poly1.contains(p1):
            return True
        else:
            return False


if __name__ == "__main__":

    triangle_1 = [[1650, 1145], [3222, 1584], [3088, 2066], [1515, 1627]]
    triangle_2 = [[3036, 1451], [3301, 1451], [3301, 1773], [3036, 1773]]
    assign_iou = ResTools.polygon_iou(triangle_1, triangle_2)
    print(assign_iou)


    # cv2.rectangle(r"", )









