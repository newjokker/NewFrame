# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import math
from shapely.geometry import Polygon, Point


# todo 分割对象中的点，应该改为点对象还是用(x,y)两个值代表的点，先改为点对象看看行不行


class SegmentObj(object):
    """一个分割对象"""

    def __init__(self, label="", points=None, shape_type="polygon", mask=None, mask_value=None):
        self.label = label
        self.points = [] if points is None else points
        self.shape_type = shape_type
        self.flags = ""
        self.line_color = None
        self.fill_color = None
        self.mask = mask
        self.mask_value = mask_value            # 当前要素在 mask 中的值（mask 中相同的 label 不同的对象有不同的值）

    def get_format_list(self):
        """获得格式化的输出"""
        return [self.label, len(self.points), self.shape_type]

    def do_print(self):
        """打印"""

        print("label : {0}".format(self.label))
        print("points count : {0}".format(len(self.points)))
        print("shape_type : {0}".format(self.shape_type))
        # print("flags : {0}".format(self.flags))
        # print("line_color : {0}".format(self.line_color))
        # print("fill_color : {0}".format(self.fill_color))

    def get_rectangle(self):
        """返回外接矩形"""
        x_min, y_min, x_max, y_max = math.inf, math.inf, -math.inf, -math.inf
        for each_point in self.points:
            each_x, each_y = each_point
            if each_x < x_min:
                x_min = each_x
            if each_x > x_max:
                x_max = each_x
            if each_y < y_min:
                y_min = each_y
            if each_y > y_max:
                y_max = each_y
        return [x_min, y_min, x_max, y_max]

    def get_area(self):
        """返回面积"""

        poly1 = Polygon(self.points).convex_hull
        return poly1.area

