# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import cv2
import copy
import time
import random
from flask import jsonify
import numpy as np
from abc import ABC
from PIL import Image
from .resBase import ResBase
from .deteObj import DeteObj
from .deteAngleObj import DeteAngleObj
from ..txkjRes.resTools import ResTools
from ..utils.JsonUtil import JsonUtil
from ..txkjRes.deteXml import parse_xml, save_to_xml, save_to_xml_wh_format
from ..utils.FileOperationUtil import FileOperationUtil
from ..utils.DecoratorUtil import DecoratorUtil
from labelme import utils


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


class PointRes(ResBase):

    def __init__(self, json_path=None, assign_img_path=None, json_dict=None):
        self._alarms = []
        self.flags = {}
        self.version = "4.4.0"
        super().__init__(assign_img_path, json_dict=json_dict)

    def __add__(self, other):
        if not isinstance(other, DeteRes):
            raise TypeError("should be DeteRes")

        res = self.deep_copy()
        for each_point_obj in other:
            if each_point_obj not in self:
                res.add_obj_2(each_point_obj)
        return res

    def __sub__(self, other):
        for each_dete_obj in other:
            self.del_dete_obj(each_dete_obj)
        return self

    def __contains__(self, item):
        if not(isinstance(item, DeteAngleObj) or isinstance(item, DeteObj)):
             raise TypeError("item should 被 DeteAngleObj or DeteObj")

        for each_point_obj in self._alarms:
            if item == each_point_obj:
                return True
        return False

    def __len__(self):
        return len(self._alarms)

    def __getitem__(self, index):
        return self._alarms[index]

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        #
        if key == 'img_path' and isinstance(value, str) and self.parse_auto:
            self._parse_img_info()
        elif key == 'json_path' and isinstance(value, str) and self.parse_auto:
            self._parse_json_file()
            pass
        elif key == 'json_dict' and isinstance(value, dict) and self.parse_auto:
            self._parse_json_str()

    @property
    def alarms(self):
        return self._alarms

    def reset_alarms(self, assign_alarms=None):
        """重置 alarms"""
        if assign_alarms is None:
            self._alarms = []
        else:
            self._alarms = assign_alarms

    def _parse_json_file(self):

        if self.json_path:
            a = JsonUtil.load_data_from_json_file(self.json_path, encoding='GBK')
        else:
            raise ValueError("* self.json_path is none")

        # parse attr
        self.version = a["version"] if "version" in a else ""
        self.width = a["imageWidth"] if "imageWidth" in a else ""
        self.height = a["imageHeight"] if "imageWidth" in a else ""
        self.file_name = a["imagePath"] if "imagePath" in a else ""
        self.image_data_bs64 = a["imageData"]

        point_index = 0
        for each_shape in a["shapes"]:
            each_shape_type = each_shape["shape_type"]           # 数据的类型 point,
            #
            if each_shape_type == 'point':
                # 解析点
                point_index += 1
                each_label = each_shape["label"]
                each_points_x, each_points_y = each_shape["points"][0]
                new_point = PointObj(each_points_x, each_points_y, each_label, assign_id=point_index)
                self.alarms.append(new_point)

    def _parse_json_str(self):
        a = self.json_dict

        # parse attr
        self.version = a["version"] if "version" in a else ""
        self.width = a["imageWidth"] if "imageWidth" in a else ""
        self.height = a["imageHeight"] if "imageWidth" in a else ""
        self.file_name = a["imagePath"] if "imagePath" in a else ""
        self.image_data_bs64 = a["imageData"]

        point_index = 0
        for each_shape in a["shapes"]:
            each_shape_type = each_shape["shape_type"]           # 数据的类型 point,
            #
            if each_shape_type == 'point':
                # 解析点
                point_index += 1
                each_label = each_shape["label"]
                each_points_x, each_points_y = each_shape["points"][0]
                new_point = PointObj(each_points_x, each_points_y, each_label, assign_id=point_index)
                self.alarms.append(new_point)

    def save_to_json_file(self, save_json_path, include_img_data=False):

        # todo 要想 labelme 能读出来，需要加上 imageData 信息，但是也支持不带 imageData 的 json 生成，可以后期使用函数进行修复，变为可读取即可

        json_info = {"version":"", "imageWidth":"", "imageHeight":"", "imagePath":"", "imageData":"", "shapes":[], "flasg":{}}

        if self.version:
            json_info["version"] = self.version
        if self.width:
            json_info["imageWidth"] = str(self.width)
        if self.height:
            json_info["imageHeight"] = str(self.height)
        if self.file_name:
            json_info["imagePath"] = self.file_name
        if self.flags:
            json_info["flags"] = self.flags
        #
        for each_shape in self._alarms:
            each_shape_info = {
                "label": each_shape.tag,
                "points": [[each_shape.x, each_shape.y]],
                "group_id": each_shape.group_id,
                "shape_type": each_shape.shape_type}
            json_info["shapes"].append(each_shape_info)

        # save img data
        if self.img_path and include_img_data:
            img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), 1)
            image_data_bs64 = utils.img_arr_to_b64(img).decode('utf-8')
            json_info["imageData"] = image_data_bs64

        # save
        JsonUtil.save_data_to_json_file(json_info, save_json_path, encoding="GBK")

    def save_to_json_str(self):
        json_info = {"version": "", "imageWidth": "", "imageHeight": "", "imagePath": "", "imageData": "", "shapes": [],
                     "flasg": {}}

        if self.version:
            json_info["version"] = self.version
        if self.image_width:
            json_info["imageWidth"] = self.image_width
        if self.image_height:
            json_info["imageHeight"] = self.image_height
        if self.img_name:
            json_info["imagePath"] = self.img_name
        if self.flags:
            json_info["flags"] = self.flags
        #
        shapes = []
        for each_shape in self._alarms:
            each_shape_info = {
                "label": each_shape.tag,
                "points": [[each_shape.x, each_shape.y]],
                "group_id": each_shape.group_id,
                "shape_type": each_shape.shape_type}
            shapes.append(each_shape_info)

        # todo 这边还需要测试和核对，
        json_dict['shapes'] = JsonUtil.save_data_to_json_str(shapes)
        return json_dict

    def draw_res(self, save_path, assign_img=None, radius=3):
        if not assign_img is None:
            img = assign_img
        else:
            img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), 1)
        #
        for each_point_obj in self:
            img = cv2.circle(img, (int(each_point_obj.x), int(each_point_obj.y)), radius, [255,255,0], thickness=2)
        cv2.imencode('.jpg', img)[1].tofile(save_path)
        return img

    def get_fzc_format(self):
        """按照防振锤模型设定的输出格式进行格式化， [tag, index, int(x1), int(y1), int(x2), int(y2), str(score)], des"""
        res_list = []
        # 遍历得到多有的
        for each_obj in self._alarms:
            res_list.append([each_obj.tag, each_obj.id, each_obj.x, each_obj.y, str(each_obj.conf), each_obj.des])
        return res_list

    def print_as_fzc_format(self):
        """按照防振锤的格式打印出来"""
        for each in self.get_fzc_format():
            print(each)

    def add_obj(self, x, y, tag, conf=-1, assign_id=-1, describe='', area=-1):
        point_res_tmp = PointObj(x=x, y=y, tag=tag, conf=conf, assign_id=assign_id, describe=describe, area=area)
        self.alarms.append(point_res_tmp)

    def add_obj_2(self, point_obj):
        self.alarms.append(point_obj)

    def deep_copy(self, copy_img=False):

        if copy_img:
            return copy.deepcopy(self)
        else:
            a = PointRes()
            a.parse_auto = False
            a.height = self.height
            a.width = self.width
            a.json_path = self.json_path
            a.img_path = self.img_path
            a.file_name = self.file_name
            a.folder = self.folder
            # img 是不进行深拷贝的，因为不会花很长的时间
            a.img = self.img
            a.json_dict = copy.deepcopy(self.json_dict)
            a.reset_alarms(copy.deepcopy(self.alarms))
            a.redis_conn_info = self.redis_conn_info
            a.img_redis_key = self.img_redis_key
            a.parse_auto = True
            return a

    def del_point_obj(self, assign_dete_obj):
        #
        for each_point_obj in copy.deepcopy(self._alarms):
            if each_point_obj == assign_dete_obj:
                # del each_dete_obj # 使用 del 删除不了
                self._alarms.remove(each_point_obj)
                # break or not
                if not del_all:
                    return

    def filter_by_tags(self, tags, update=True):

        if tags:
            res = self.deep_copy()
            res.reset_alarms()
            for each_point_obj in self:
                if each_point_obj.tag in tags:
                    res.add_obj_2(each_point_obj)

            if update:
                self.reset_alarms(res.alarms)
            return res









