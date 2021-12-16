# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import copy
import os
import cv2
import math


# todo PointObj 和 deteObj 之间的相互转换（1）当已知 pointObj 面积的时候，对应的 deteRes 按照正方形计算出他的边长 （2）deteObj 的中心点作为 pointObj 的位置信息，area 也对应过去即可


class DeteObj(object):
    """检测结果的一个检测对象，就是一个矩形框对应的信息"""

    def __init__(self, x1=None, y1=None, x2=None, y2=None, tag="", conf=-1, assign_id=-1, describe:str=''):
        """(x1,y1), (x2,y2) 左下角右上角"""
        self.conf = conf
        self.tag = tag
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.id = assign_id
        # 描述信息，用于接纳非标准信息
        self.des = describe
        # crop 下来的小图保存的位置
        self.crop_path = ""

    def __eq__(self, other):
        """等于"""

        # 类型不同返回 false
        if not isinstance(other, DeteObj):
            return False

        if self.x1 == other.x1 and self.x2 == other.x2 and self.y1 == other.y1 and self.y2 == other.y2 and self.tag == other.tag:
            return True
        else:
            return False

    def approximate(self, other, ignore_tag=False, iou_th=0.8):
        """近似"""
        pass

    def do_offset(self, offset_x, offset_y):
        """对结果进行偏移"""
        self.x1 += offset_x
        self.x2 += offset_x
        self.y1 += offset_y
        self.y2 += offset_y

    def do_augment(self, augment_parameter, width, height, is_relative=True):
        """对框进行扩展，这边传入的绝对比例，或者相对"""
        region_width = int(self.x2 - self.x1)
        region_height = int(self.y2 - self.y1)
        #
        if is_relative:
            new_x_min = self.x1 - int(region_width * augment_parameter[0])
            new_x_max = self.x2 + int(region_width * augment_parameter[1])
            new_y_min = self.y1 - int(region_height * augment_parameter[2])
            new_y_max = self.y2 + int(region_height * augment_parameter[3])
        else:
            new_x_min = self.x1 - int(augment_parameter[0])
            new_x_max = self.x2 + int(augment_parameter[1])
            new_y_min = self.y1 - int(augment_parameter[2])
            new_y_max = self.y2 + int(augment_parameter[3])
        #
        new_x_min = max(0, new_x_min)
        new_y_min = max(0, new_y_min)
        new_x_max = min(width - 1, new_x_max)
        new_y_max = min(height-1, new_y_max)
        #
        self.x1 = new_x_min
        self.x2 = new_x_max
        self.y1 = new_y_min
        self.y2 = new_y_max

    def get_rectangle(self):
        """获取矩形范围"""
        return [self.x1, self.y1, self.x2, self.y2]

    def get_center_point(self):
        """得到中心点坐标"""
        return float(self.x1+self.x2)/2, float(self.y1+self.y2)/2

    def get_format_list(self):
        """得到标准化的 list 主要用于打印"""
        return [str(self.tag), int(self.x1), int(self.y1), int(self.x2), int(self.y2), format(float(self.conf), '.4f')]

    def get_area(self):
        """返回面积，面积大小按照像素个数进行统计"""
        return int(self.x2 - self.x1) * int(self.y2 - self.y1)

    def get_points(self):
        """返回四边形顺序上的四个点"""
        return [[self.x1,self.y1], [self.x2,self.y1], [self.x2,self.y2], [self.x1, self.y2]]

    def get_point_obj(self):
        x, y = self.get_center_point()
        if self.des:
            describe = self.des
        else:
            describe = "width:{0},height:{1}".format(x, y)
        res = PointObj(x=x, y=y, conf=self.conf, tag=self.tag, describe=describe, assign_id=self.id, area=self.get_area())
        return res

    def format_check(self):
        """类型检查和调整"""
        self.conf = float(self.conf)
        self.tag = str(self.tag)
        self.x1 = int(self.x1)
        self.y1 = int(self.y1)
        self.x2 = int(self.x2)
        self.y2 = int(self.y2)

    def deep_copy(self):
        """返回深拷贝对象"""
        return copy.deepcopy(self)

    def get_name_str(self, assign_loc=None):
        """信息保存为文件名"""
        if assign_loc:
            name_str = "[{0},{1},{2},{3},{4},{5},{6}]".format(assign_loc[0], assign_loc[1], assign_loc[2], assign_loc[3], "'" + self.tag + "'", self.conf, self.id)
        else:
            name_str = "[{0},{1},{2},{3},{4},{5},{6}]".format(self.x1, self.y1, self.x2, self.y2, "'" + self.tag + "'", self.conf, self.id)
        return name_str

    def load_from_name_str(self, name_str):
        """从文件名获取信息"""
        self.x1, self.y1, self.x2, self.y2, self.tag, self.conf, self.id = eval(name_str)

    def get_crop_img(self, RGB=True, assign_shape_min=False):
        """根据指定的 deteObj 读取裁剪的 小图"""

        if not os.path.exists(self.crop_path):
            return None

        im_array = cv2.imread(self.crop_path)

        if assign_shape_min:
            w, h = im_array.shape[:2]
            ratio = assign_shape_min/min(w, h)
            im_array = cv2.resize(im_array, (int(ratio*w), int(ratio*h)))

        if RGB:
            return cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        else:
            return im_array

    def del_crop_img(self):
        """删除裁剪下来的缓存文件"""
        if os.path.exists(self.crop_path):
            os.remove(self.crop_path)
            self.crop_path = ''


class PointObj(object):

    def __init__(self, x, y, tag, conf=-1, assign_id=-1, describe='', area=-1):
        self.x = x
        self.y = y
        self.tag = tag
        self.area = area  # 点是可以有面积的，当这个点是具体目标的抽象的时候，此时面积就是点的一个属性
        self.conf = conf
        self.id = assign_id
        self.des = describe
        self.shape_type = 'point'
        self.group_id = None

    def __eq__(self, other):
        # 类型不同返回 false
        if not isinstance(other, PointObj):
            return False

        if self.x == other.x and self.y == other.y and self.tag == other.tag:
            return True
        else:
            return False

    def do_offset(self, offset_x, offset_y):
        self.x += offset_x
        self.y += offset_y

    def deep_copy(self):
        return copy.deepcopy(self)

    def get_name_str(self):
        name_str = "[{0},{1},{2},{3},{4}]".format(self.x, self.y, "'" + self.tag + "'", self.conf, self.id)
        return name_str

    def load_from_name_str(self, name_str):
        self.x, self.y, self.tag, self.conf, self.id = eval(name_str)

    def get_dete_obj(self, assign_wh=None):
        """转换为 deteobj"""
        # arae 必须有值，才能计算出长宽，否则需要指定长宽
        if self.area == -1 and assign_wh is None:
            raise ValueError("* pointObj's area is empty")
        # 计算出目标的长宽
        elif assign_wh is not None:
            width, height = assign_wh
        else:
            width = height = math.sqrt(self.area)
        #
        x1 = self.x - (width/2)
        x2 = self.x + (width/2)
        y1 = self.y - (height/2)
        y2 = self.y + (height/2)
        res = DeteObj(x1=x1, x2=x2, y1=y1, y2=y2, conf=self.conf, tag=self.tag, assign_id=self.id, describe=self.des)
        return res


if __name__ == "__main__":

    a = DeteObj(10,10,30,30,'ok_good')
    # b = a.get_name_str()
    # print(b)
    # a.load_from_name_str(b)
    # print(a.get_format_list())

    b= a.get_point_obj()

    print(b.get_name_str())
    print(b.des)
    print(b.area)

    b.get_dete_obj()

    print(b.get_name_str())
    print(b.des)





