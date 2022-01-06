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

"""
'__abstractmethods__',
 '__contains__',
 '__getitem__',
 '__init__',
 '__len__',
 '__setattr__',
 '_parse_img_info',
 '_parse_img_info_from_redis',
 '_parse_json_info',
 '_parse_xml_info',
 # 
 'add_angle_obj',                       # [*]增加旋转框
 'add_obj',                             # [*]增加矩形框
 'add_obj_2',                           # 增加旋转框或者矩形框或者其他什么框
 'angle_obj_to_obj',                    # 旋转框转矩形框
 'count_tags',                          # 统计标签个数
 'crop_and_save',                       # 矩形框保存小截图
 'crop_angle_and_save',                 # 旋转框保存小截图
 'crop_dete_obj',                       # [*]将正框裁剪到指定文件夹，裁剪路径会记录在 obj 的 crop_path 路径下
 'crop_with_xml',                       # 裁剪正框并生成对应的 xml 
 'deep_copy',                           # [*]深拷贝
 'del_dete_obj',                        # [*]删除指定 dete obj
 'del_sub_img_from_crop',               # [*]删除保存的截图，DeteObj crop_path 属性记录了截图的位置
 'do_augment',                          # 框的扩增（上下左右）
 'do_func',                             # 做指定的操作
 'do_nms',                              # [*]对框之间做 nms 
 'do_nms_center_point',                 # 对框之间做 center nms
 'do_nms_in_assign_tags',               # 在指定几种 tag 之间做 nms 
 'draw_dete_res',                       # [*]画图
 'filter_by_area',                      # 根据框的绝对面积进行筛除
 'filter_by_area_ratio',                # 根据框的相对面积进行筛除
 'filter_by_conf',                      # [*]根据框的置信度进行筛除
 'filter_by_func',                      # 使用指定方法对框进行筛除
 'filter_by_mask',                      # 使用掩膜（掩膜内，掩膜外）对框进行筛除
 'filter_by_dete_res_mask',             # 将指定 deteres 的每一个对象做一个 mask 进行掩膜操作
 'filter_by_tags',                      # [*]根据标签类型对框进行筛除
 'filter_by_topn',                      # 对deteobj 大小进行排序，根据前 nn 个目标的 1/2 对目标进行过滤
 'get_crop_name_by_id',                 # 获取指定 id 对应的裁剪名（考虑删除该方法） 
 'get_dete_obj_by_id',                  # 获取指定 id 对应的第一个 dete obj
 'get_dete_obj_list_by_func',           # 获取指定方法找到的 dete obj list
 'get_dete_obj_list_by_id',             # 获取等于指定id的 dete obj list
 'get_dete_obj_list_by_tag',            # [*]获取等于指定tag的 dete obj list
 'get_fzc_format',                      # 将所有 dete obj 按照一定的样式返回
 'get_id_list',                         # 获取所有的标签 id list (考虑删除函数)
 'get_img_array',                       # [*]获取图像的矩阵
 'get_img_from_resdis',                 # 从redis中获取图像矩阵
 'get_result_construction',             # 获取标准返回给 java 的结果
 'get_return_jsonify',                  # [*]获取标准返回结果（子模型的返回）
 'get_sub_img_by_dete_obj',             # [*]拿到指定 dete obj 对应的图像矩阵
 'get_sub_img_by_dete_obj_from_crop',   # [*]如果之前进行过小图裁剪，可以通过这个方法读取 dete obj 对应的小图，返回矩阵
 'get_sub_img_by_id',                   # 找到指定 id 第一个小图，返回框范围矩阵
 'has_tag',                             # 是否存在指定标签
 'offset',                              # [*]对多有框进行偏移
 'parse_txt_info',                      # 解析 txt 信息（之前准备使用 txt 存储 xml 中的信息，考虑删除该方法）
 'print_as_fzc_format',                 # 对框的标准打印（考虑换一个名字）
 'refresh_obj_id',                      # 刷新框（dete obj）的id
 'reset_alarms',                        # 重设框的信息
 'save_assign_range',                   # 保存图像指定范围内的信息（图像矩阵，框）
 'save_to_json',                        # [*]返回 json str
 'save_to_txt',                         # 保存为 txt
 'save_to_xml',                         # [*]保存为 xml
 'set_img_to_redis',                    # 将图像存储到redis中
 'update_tags',                         # 更新标签
 # ------------------------
 'intersection',                        # 交集
 'intersection_update',                 # 交集，更新
 'union',                               # 并集
 'difference',                          # 差集
 'difference_update',                   # 差集，更新
 'issubset',                            # 子集
 'isupperset',                          # 超集
 # ------------------------
 '_log',                                # 日志对象（准备自动记录日志使用，还未实现）
 'width',                               # 图像宽度
 'height',                              # 图像高度
 'xml_path'                             # 图像的 xml path
 'file_name',                           # 文件名
 'alarms',                              # dete obj 列表
 'img',                                 # 图像对象（PIL）
 'folder',                              # 文件夹路径
 'json_dict',                           # dete obj 对应的 json str
 'img_path',                            # 图像路径
 'img_redis_key',                       # 存储在redis中的图像的key
 'redis_conn',                          # redis 操作对象
 'redis_conn_info',                     # redis 中存储的图像信息
 'parse_auto',                          # 是否自动解析信息（例，当给对象的 img_path 属性赋值的时候，是否自动读取图像的信息，并保存在对象中）
"""


# todo 将 self.img 从 PIL 结构保存为 ndarry 结构，使用空间换取时间

# todo 直接对图像矩阵进行处理，而不是对 PIL 对象进行处理，看看是不是快很多

# todo 查看图片三个波段反了是因为，用的 cv2.imwrite() 保存的图片

# todo crop_dete_obj 等于 crop_and_save 不用把同样的功能实现两遍

# fixme 获取小图截图范围的时候需要看看是不是 RGB 三通道是反的，是不是需要处理一下



class DeteRes(ResBase, ABC):
    """检测结果"""

    def __init__(self, xml_path=None, assign_img_path=None, json_dict=None, log=None, redis_conn_info=None, img_redis_key=None):
        # 子类新方法需要放在前面
        self._alarms = []
        self._log = log
        super().__init__(xml_path, assign_img_path, json_dict, redis_conn_info=redis_conn_info, img_redis_key=img_redis_key)

    def __contains__(self, item):
        """是否包含元素"""

        if not(isinstance(item, DeteAngleObj) or isinstance(item, DeteObj)):
             raise TypeError("item should 被 DeteAngleObj or DeteObj")

        for each_dete_obj in self._alarms:
            if item == each_dete_obj:
                return True

        return False

    def __add__(self, other):
        """DeteRes之间进行相加"""

        if not isinstance(other, DeteRes):
            raise TypeError("should be DeteRes")

        res = self.deep_copy()
        for each_dete_obj in other:
            # 不包含这个元素的时候进行添加
            if each_dete_obj not in self:
                res.add_obj_2(each_dete_obj)
        return res

    def __sub__(self, other):
        """DeteRes之间相减"""
        res = self.deep_copy()
        for each_dete_obj in other:
            res.del_dete_obj(each_dete_obj)
        return res

    def __len__(self):
        """返回要素的个数"""
        return len(self._alarms)

    def __getitem__(self, index):
        """按照 index 取对应的对象"""
        return self._alarms[index]

    def __setattr__(self, key, value):
        """设置属性后执行对应"""
        object.__setattr__(self, key, value)
        #
        if key == 'img_path' and isinstance(value, str) and self.parse_auto:
            self._parse_img_info()
        elif key == 'xml_path' and isinstance(value, str) and self.parse_auto:
            self._parse_xml_info()
        elif key == 'json_dict' and isinstance(value, dict) and self.parse_auto:
            self._parse_json_info()

    # ------------------------------------------ transform -------------------------------------------------------------
    @property
    def alarms(self):
        """获取属性自动进行排序"""
        # return sorted(self._alarms, key=lambda x:x.id)
        return self._alarms

    def _parse_xml_info(self):
        """解析 xml 中存储的检测结果"""
        xml_info = parse_xml(self.xml_path)
        #
        if 'size' in xml_info:
            if 'height' in xml_info['size']:
                self.height = float(xml_info['size']['height'])
            if 'width' in xml_info['size']:
                self.width = float(xml_info['size']['width'])
        #
        if 'filename' in xml_info:
            self.file_name = xml_info['filename']
        #
        if 'path' in xml_info:
            self.img_path = xml_info['path']

        if 'folder' in xml_info:
            self.folder = xml_info['folder']

        if "des" in xml_info:
            self.des = xml_info["des"]

        # 解析 object 信息
        for each_obj in xml_info['object']:
            # bndbox
            if 'bndbox' in each_obj:
                bndbox = each_obj['bndbox']

                if not bndbox:
                    break

                # ------------------------------------------------------------------------------------------------------
                # fixme 恶心的代码，在同一后进行删除

                if 'xmin' in bndbox:
                    x_min, x_max, y_min, y_max = int(bndbox['xmin']), int(bndbox['xmax']), int(bndbox['ymin']), int(bndbox['ymax'])
                elif 'xMin' in bndbox:
                    x_min, x_max, y_min, y_max = int(bndbox['xMin']), int(bndbox['xMax']), int(bndbox['yMin']), int(bndbox['yMax'])
                else:
                    continue
                # ------------------------------------------------------------------------------------------------------

                if 'prob' not in each_obj: each_obj['prob'] = -1
                if 'id' not in each_obj: each_obj['id'] = -1
                if 'des' not in each_obj: each_obj['des'] = ''
                if 'crop_path' not in each_obj: each_obj['crop_path'] = ''
                if each_obj['id'] in ['None', None]: each_obj['id'] = -1
                each_dete_obj = DeteObj(x1=x_min, x2=x_max, y1=y_min, y2=y_max, tag=each_obj['name'], conf=float(each_obj['prob']), assign_id=int(each_obj['id']), describe=each_obj['des'])
                each_dete_obj.crop_path = each_obj['crop_path']
                self.add_obj_2(each_dete_obj)
                # self.add_obj(x1=x_min, x2=x_max, y1=y_min, y2=y_max, tag=each_obj['name'], conf=float(each_obj['prob']), assign_id=int(each_obj['id']), describe=each_obj['des'])
            # robndbox
            if 'robndbox' in each_obj:
                bndbox = each_obj['robndbox']
                cx, cy, w, h, angle = float(bndbox['cx']), float(bndbox['cy']), float(bndbox['w']), float(bndbox['h']), float(bndbox['angle'])
                if 'prob' not in each_obj: each_obj['prob'] = -1
                if 'id' not in each_obj : each_obj['id'] = -1
                if 'des' not in each_obj : each_obj['des'] = ''
                if 'crop_path' not in each_obj : each_obj['crop_path'] = ''
                # fixme 这块要好好修正一下，这边应为要改 bug 暂时这么写的
                if each_obj['id'] in ['None', None] : each_obj['id'] = -1
                each_dete_obj = DeteAngleObj(cx, cy, w, h, angle, tag=each_obj['name'], conf=each_obj['prob'], assign_id=each_obj['id'], describe=each_obj['des'])
                each_dete_obj.crop_path = each_obj['crop_path']
                self.add_obj_2(each_dete_obj)
                # self.add_angle_obj(cx, cy, w, h, angle, tag=each_obj['name'], conf=each_obj['prob'], assign_id=each_obj['id'], describe=each_obj['des'])

    def _parse_json_info(self):
        """解析 json 信息"""

        json_info = self.json_dict

        if 'size' in json_info:
            json_info['size'] = JsonUtil.load_data_from_json_str(json_info['size'])
            if 'height' in json_info['size']:
                self.height = float(json_info['size']['height'])
            if 'width' in json_info['size']:
                self.width = float(json_info['size']['width'])
        #
        if 'filename' in json_info:
            self.file_name = json_info['filename']

        if 'path' in json_info:
            self.img_path = json_info['path']

        if 'folder' in json_info:
            self.folder = json_info['folder']

        if 'des' in json_info:
            self.folder = json_info['des']

        # 解析 object 信息
        if 'object' in json_info:
            for each_obj in JsonUtil.load_data_from_json_str(json_info['object']):
                each_obj = JsonUtil.load_data_from_json_str(each_obj)
                # bndbox
                if 'bndbox' in each_obj:
                    bndbox = each_obj['bndbox']
                    x_min, x_max, y_min, y_max = int(bndbox['xmin']), int(bndbox['xmax']), int(bndbox['ymin']), int(bndbox['ymax'])
                    if 'prob' not in each_obj: each_obj['prob'] = -1
                    if 'id' not in each_obj: each_obj['id'] = -1
                    if 'des' not in each_obj: each_obj['des'] = ''
                    if 'crop_path' not in each_obj: each_obj['crop_path'] = ''
                    each_dete_obj = DeteObj(x1=x_min, x2=x_max, y1=y_min, y2=y_max, tag=each_obj['name'], conf=float(each_obj['prob']), assign_id=int(each_obj['id']), describe=str(each_obj['des']))
                    each_dete_obj.crop_path = each_obj['crop_path']
                    self.add_obj_2(each_dete_obj)
                    # self.add_obj(x1=x_min, x2=x_max, y1=y_min, y2=y_max, tag=each_obj['name'], conf=float(each_obj['prob']), assign_id=int(each_obj['id']), describe=str(each_obj['des']))
                # robndbox
                if 'robndbox' in each_obj:
                    bndbox = each_obj['robndbox']
                    cx, cy, w, h, angle = float(bndbox['cx']), float(bndbox['cy']), float(bndbox['w']), float(bndbox['h']), float(bndbox['angle'])
                    if 'prob' not in each_obj: each_obj['prob'] = -1
                    if 'id' not in each_obj: each_obj['id'] = -1
                    if 'des' not in each_obj: each_obj['des'] = -1
                    if 'crop_path' not in each_obj: each_obj['crop_path'] = ''
                    each_dete_obj = DeteAngleObj(cx, cy, w, h, angle, tag=each_obj['name'], conf=float(each_obj['prob']),assign_id=int(each_obj['id']), describe=str(each_obj['des']))
                    each_dete_obj.crop_path = each_obj['crop_path']
                    self.add_obj_2(each_dete_obj)
                    # self.add_angle_obj(cx, cy, w, h, angle, tag=each_obj['name'], conf=float(each_obj['prob']),assign_id=int(each_obj['id']), describe=str(each_obj['des']))

    def save_to_xml(self, save_path, assign_alarms=None, format='normal'):
        """保存为 xml 文件"""
        xml_info = {'size': {'height': str(int(self.height)), 'width': str(int(self.width)), 'depth': '3'},
                    'filename': self.file_name, 'path': self.img_path, 'object': [], 'folder': self.folder,
                    'segmented': "", 'source': "", "des": ""}

        if assign_alarms is None:
            alarms = self._alarms
        else:
            alarms = assign_alarms
        #
        for each_dete_obj in alarms:
            # bndbox
            if isinstance(each_dete_obj, DeteObj):
                each_obj = {'name': each_dete_obj.tag, 'prob': str(each_dete_obj.conf), 'id':str(each_dete_obj.id), 'des':str(each_dete_obj.des),'crop_path':str(each_dete_obj.crop_path),
                            'bndbox': {'xmin': str(int(each_dete_obj.x1)), 'xmax': str(int(each_dete_obj.x2)),
                                       'ymin': str(int(each_dete_obj.y1)), 'ymax': str(int(each_dete_obj.y2))}}
                xml_info['object'].append(each_obj)
            # robndbox
            elif isinstance(each_dete_obj, DeteAngleObj):
                each_obj = {'name': each_dete_obj.tag, 'prob': str(each_dete_obj.conf), 'id': str(int(each_dete_obj.id)), 'des':str(each_dete_obj.des),'crop_path':str(each_dete_obj.crop_path),
                            'robndbox': {'cx': str(each_dete_obj.cx), 'cy': str(each_dete_obj.cy),
                                         'w': str(each_dete_obj.w), 'h': str(each_dete_obj.h),'angle': str(each_dete_obj.angle)}}
                xml_info['object'].append(each_obj)

        # 保存为 xml
        if format == 'normal':
            save_to_xml(xml_info, xml_path=save_path)
        elif format == 'wuhan':
            save_to_xml_wh_format(xml_info, xml_path=save_path)

    def save_to_json(self, assign_alarms=None):
        """转为 json 结构"""

        json_dict = {'size': JsonUtil.save_data_to_json_str({'height': int(self.height), 'width': int(self.width), 'depth': '3'}),
                    'filename': self.file_name, 'path': self.img_path, 'object': [], 'folder': self.folder,
                    'segmented': "", 'source': "", "des":self.des}
        # 可以指定输出的 alarms
        if assign_alarms is None:
            alarms = self._alarms
        else:
            alarms = assign_alarms
        #
        json_object = []
        for each_dete_obj in alarms:
            # bndbox
            if isinstance(each_dete_obj, DeteObj):
                each_obj = {'name': each_dete_obj.tag, 'prob': float(each_dete_obj.conf), 'id':int(each_dete_obj.id), 'des':str(each_dete_obj.des), 'crop_path':str(each_dete_obj.crop_path),
                            'bndbox': {'xmin': int(each_dete_obj.x1), 'xmax': int(each_dete_obj.x2),
                                       'ymin': int(each_dete_obj.y1), 'ymax': int(each_dete_obj.y2)}}
                json_object.append(JsonUtil.save_data_to_json_str(each_obj))
            # robndbox
            elif isinstance(each_dete_obj, DeteAngleObj):
                each_obj = {'name': each_dete_obj.tag, 'prob': str(each_dete_obj.conf), 'id': str(each_dete_obj.id), 'des':str(each_dete_obj.des), 'crop_path':str(each_dete_obj.crop_path),
                            'robndbox': {'cx': float(each_dete_obj.cx), 'cy': float(each_dete_obj.cy),
                                         'w': float(each_dete_obj.w), 'h': float(each_dete_obj.h),
                                         'angle': float(each_dete_obj.angle)}}
                json_object.append(JsonUtil.save_data_to_json_str(each_obj))
        json_dict['object'] = JsonUtil.save_data_to_json_str(json_object)
        return json_dict

    def save_to_txt(self, txt_path):
        """label img 中会将标注信息转为 txt 进行保存"""
        # todo 会生成两个文件 （1）classes.txt 存放类别信息 （2）文件名.txt 存放标注信息，tag mx my w h , mx my 为中心点坐标
        pass

    # @DecoratorUtil.time_this
    def crop_dete_obj(self, save_dir, augment_parameter=None, method=None, exclude_tag_list=None, split_by_tag=False, include_tag_list=None, assign_img_name=None):
        """将指定的类型的结果进行保存，可以只保存指定的类型，命名使用标准化的名字 fine_name + tag + index, 可指定是否对结果进行重采样，或做特定的转换，只要传入转换函数
        * augment_parameter = [0.5, 0.5, 0.2, 0.2]
        """
        # fixme 存储 crop 存的文件夹，

        if not self.img:
            raise ValueError ("need img_path or img")

        #
        if assign_img_name is not None:
            img_name = assign_img_name
        else:
            if self.file_name:
                img_name = os.path.split(self.file_name)[1][:-4]
            elif self.img_path is not None :
                img_name = os.path.split(self.img_path)[1][:-4]
            else:
                raise ValueError("need self.img_path or assign_img_name")

        tag_count_dict = {}
        #
        for each_obj in self._alarms:
            # 只支持正框的裁切
            if not isinstance(each_obj, DeteObj):
                continue
            # 截图的区域
            bndbox = [each_obj.x1, each_obj.y1, each_obj.x2, each_obj.y2]
            # 排除掉不需要保存的 tag
            if include_tag_list is not None:
                if each_obj.tag not in include_tag_list:
                    continue

            if not exclude_tag_list is None:
                if each_obj.tag in exclude_tag_list:
                    continue

            # 计算这是当前 tag 的第几个图片
            if each_obj.tag not in tag_count_dict:
                tag_count_dict[each_obj.tag] = 0
            else:
                tag_count_dict[each_obj.tag] += 1
            # 图片扩展
            if augment_parameter is not None:
                bndbox = ResTools.region_augment(bndbox, [self.width, self.height], augment_parameter=augment_parameter)

            # 为了区分哪里是最新加上去的，使用特殊符号 -+- 用于标志
            if split_by_tag is True:
                each_save_dir = os.path.join(save_dir, each_obj.tag)
                if not os.path.exists(each_save_dir):
                    os.makedirs(each_save_dir)
            else:
                each_save_dir = save_dir

            # fixme 图像范围进行扩展，但是标注的范围不进行扩展，这边要注意
            each_name_str = each_obj.get_name_str()
            each_save_path = os.path.join(each_save_dir, '{0}-+-{1}.jpg'.format(img_name, each_name_str))
            #
            each_obj.crop_path = each_save_path
            #
            each_crop = self.img.crop(bndbox)
            # 保存截图
            # each_crop.save(each_save_path, quality=95)
            each_crop.save(each_save_path)

    @DecoratorUtil.time_this
    def crop_dete_obj_new(self, save_dir, augment_parameter=None, method=None, exclude_tag_list=None, split_by_tag=False, include_tag_list=None, assign_img_name=None):
        """将指定的类型的结果进行保存，可以只保存指定的类型，命名使用标准化的名字 fine_name + tag + index, 可指定是否对结果进行重采样，或做特定的转换，只要传入转换函数
        * augment_parameter = [0.5, 0.5, 0.2, 0.2]
        """
        # fixme 存储 crop 存的文件夹，

        if self.img_ndarry is None:
            self.img_ndarry = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), 1)
        #
        if assign_img_name is not None:
            img_name = assign_img_name
        else:
            if self.file_name:
                img_name = os.path.split(self.file_name)[1][:-4]
            elif self.img_path is not None :
                img_name = os.path.split(self.img_path)[1][:-4]
            else:
                raise ValueError("need self.img_path or assign_img_name")

        tag_count_dict = {}
        #
        for each_obj in self._alarms:
            # 只支持正框的裁切
            if not isinstance(each_obj, DeteObj):
                continue
            # 截图的区域
            bndbox = [each_obj.x1, each_obj.y1, each_obj.x2, each_obj.y2]
            # 排除掉不需要保存的 tag
            if include_tag_list is not None:
                if each_obj.tag not in include_tag_list:
                    continue

            if not exclude_tag_list is None:
                if each_obj.tag in exclude_tag_list:
                    continue

            # 计算这是当前 tag 的第几个图片
            if each_obj.tag not in tag_count_dict:
                tag_count_dict[each_obj.tag] = 0
            else:
                tag_count_dict[each_obj.tag] += 1
            # 图片扩展
            if augment_parameter is not None:
                bndbox = ResTools.region_augment(bndbox, [self.width, self.height], augment_parameter=augment_parameter)

            # 为了区分哪里是最新加上去的，使用特殊符号 -+- 用于标志
            if split_by_tag is True:
                each_save_dir = os.path.join(save_dir, each_obj.tag)
                if not os.path.exists(each_save_dir):
                    os.makedirs(each_save_dir)
            else:
                each_save_dir = save_dir

            # fixme 图像范围进行扩展，但是标注的范围不进行扩展，这边要注意
            each_name_str = each_obj.get_name_str()
            each_save_path = os.path.join(each_save_dir, '{0}-+-{1}.jpg'.format(img_name, each_name_str))
            #
            each_obj.crop_path = each_save_path
            #
            # each_crop = self.img.crop(bndbox)

            each_crop = self.img_ndarry[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2], :]
            cv2.imencode(each_save_path, each_crop)[1].tofile(each_save_path)

            # 保存截图
            # each_crop.save(each_save_path, quality=95)
            # each_crop.save(each_save_path)

    def _parse_txt_info(self, classes_path, record_path):
        """解析 txt 信息"""
        # todo txt 信息中不包含图像的大小，波段数等信息，保存和读取 txt 标注的信息比较鸡肋
        pass

    # --------------------------------------------- id -----------------------------------------------------------------

    def get_dete_obj_by_id(self, assign_id):
        """获取第一个 id 对应的 deteObj 对象"""
        for each_dete_obj in self._alarms:
            if int(each_dete_obj.id) == int(assign_id):
                return each_dete_obj
        return None

    def get_id_list(self):
        """获取要素 id list，有时候会过滤掉一些 id 这时候按照 id 寻找就会有问题"""
        id_set = set()
        for each_dete_obj in self._alarms:
            id_set.add(each_dete_obj.id)
        return list(id_set)

    def refresh_obj_id(self):
        """跟新要素的 id，重新排列"""
        index = 0
        for each_dete_obj in self._alarms:
            each_dete_obj.id = index
            index += 1

    def get_crop_name_by_id(self, assign_id):
        """根据文件的ID得到文件裁剪后的名字"""
        img_name = os.path.split(self.img_path)[1]
        dete_obj = self.get_dete_obj_by_id(assign_id)
        name_str = dete_obj.get_name_str()
        crop_name = '{0}-+-{1}.jpg'.format(img_name, name_str)
        return crop_name

    def get_sub_img_by_id(self, assign_id, augment_parameter=None, RGB=True, assign_shape_min=False):
        """根据指定 id 得到小图的矩阵数据"""
        assign_dete_obj = self.get_dete_obj_by_id(assign_id=assign_id)
        return self.get_sub_img_by_dete_obj(assign_dete_obj, augment_parameter, RGB=RGB, assign_shape_min=assign_shape_min)

    # @DecoratorUtil.time_this
    def get_sub_img_by_dete_obj(self, assign_dete_obj, augment_parameter=None, RGB=True, assign_shape_min=False):
        """根据指定的 deteObj """

        # 如果没有读取 img
        if not self.img:
            raise ValueError ("need img_path or img")

        if isinstance(assign_dete_obj, DeteObj):
            if augment_parameter is None:
                crop_range = [assign_dete_obj.x1, assign_dete_obj.y1, assign_dete_obj.x2, assign_dete_obj.y2]
            else:
                crop_range = [assign_dete_obj.x1, assign_dete_obj.y1, assign_dete_obj.x2, assign_dete_obj.y2]
                crop_range = ResTools.region_augment(crop_range, [self.width, self.height], augment_parameter=augment_parameter)
            img_crop = self.img.crop(crop_range)
        elif isinstance(assign_dete_obj, DeteAngleObj):
            if augment_parameter is None:
                crop_array = ResTools.crop_angle_rect(np.array(self.img), ((assign_dete_obj.cx, assign_dete_obj.cy), (assign_dete_obj.w, assign_dete_obj.h), assign_dete_obj.angle))
            else:
                w = assign_dete_obj.w * (1+augment_parameter[0])
                h = assign_dete_obj.h * (1+augment_parameter[1])
                crop_array = ResTools.crop_angle_rect(np.array(self.img), ((assign_dete_obj.cx, assign_dete_obj.cy), (w, h), assign_dete_obj.angle))
            # BGR -> RGB
            img_crop = Image.fromarray(crop_array)
        else:
            raise ValueError("not support assign_dete_obj's type : ".format(type(assign_dete_obj)))

        # change size
        if assign_shape_min:
            w, h = img_crop.width, img_crop.height
            ratio = assign_shape_min/min(w, h)
            img_crop = img_crop.resize((int(ratio*w), int(ratio*h)))

        # Image --> array
        im_array = np.array(img_crop)
        # change chanel order
        if RGB:
            return im_array
        else:
            return cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)

    @DecoratorUtil.time_this
    def get_sub_img_by_dete_obj_new(self, assign_dete_obj, augment_parameter=None, RGB=True, assign_shape_min=False):
        """根据指定的 deteObj """

        if self.img_ndarry is None:
            img_ndarry = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), 1)
            # self.img_ndarry = cv2.cvtColor(img_ndarry, cv2.COLOR_RGB2BGR)

        if augment_parameter is None:
            crop_range = [assign_dete_obj.x1, assign_dete_obj.y1, assign_dete_obj.x2, assign_dete_obj.y2]
        else:
            crop_range = [assign_dete_obj.x1, assign_dete_obj.y1, assign_dete_obj.x2, assign_dete_obj.y2]
            crop_range = ResTools.region_augment(crop_range, [self.width, self.height], augment_parameter=augment_parameter)

        img_crop = self.img_ndarry[crop_range[1]: crop_range[3], crop_range[0]: crop_range[2], :]

        # # change size
        # if assign_shape_min:
        #     w, h = img_crop.shape[:2]
        #     ratio = assign_shape_min/min(w, h)
        #     img_crop = img_crop.resize((int(ratio*w), int(ratio*h)))

        if RGB:
            return img_crop
        else:
            return cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)

    @staticmethod
    def get_sub_img_by_dete_obj_from_crop(assign_dete_obj, RGB=True, assign_shape_min=False):
        """根据指定的 deteObj 读取裁剪的 小图"""
        return assign_dete_obj.get_crop_img(RGB=RGB, assign_shape_min=assign_shape_min)

    def del_sub_img_from_crop(self):
        """删除裁剪的缓存文件"""
        for each_dete_obj in self:
            each_dete_obj.del_crop_img()

    @DecoratorUtil.time_this
    def get_img_array_new(self, RGB=True):
        """获取self.img对应的矩阵信息"""

        if self.img_ndarry is None:
            self.img_ndarry = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), 1)

        if RGB:
            return cv2.cvtColor(self.img_ndarry, cv2.COLOR_RGB2BGR)
        else:
            return self.img_ndarry

    @DecoratorUtil.time_this
    def get_img_array(self, RGB=True):
        """获取self.img对应的矩阵信息"""
        if not self.img:
            raise ValueError ("need img_path or img")

        if RGB:
            return np.array(self.img)
        else:
            return cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)

    def get_dete_obj_list_by_id(self, assign_id, is_deep_copy=False):
        """获取所有 id 对应的 deteObj 对象，可以指定是否执行深拷贝"""
        res = []
        for each_dete_obj in self._alarms:
            if int(each_dete_obj.id) == int(assign_id):
                if is_deep_copy:
                    res.append(each_dete_obj.deep_copy())
                else:
                    res.append(each_dete_obj)
        return res

    def get_dete_obj_list_by_tag(self, need_tags, is_deep_copy=False):
        """获取所有 id 对应的 deteObj 对象，可以指定是否执行深拷贝"""
        res = []
        for each_dete_obj in self._alarms:
            if each_dete_obj.tag in need_tags:
                if is_deep_copy:
                    res.append(each_dete_obj.deep_copy())
                else:
                    res.append(each_dete_obj)
        return res

    # ------------------------------------------------ get -------------------------------------------------------------

    def add_obj(self, x1, y1, x2, y2, tag, conf=-1, assign_id=-1, describe=''):
        """快速增加一个检测框要素"""
        one_dete_obj = DeteObj(x1=x1, y1=y1, x2=x2, y2=y2, tag=tag, conf=conf, assign_id=assign_id, describe=describe)
        self._alarms.append(one_dete_obj)

    def add_angle_obj(self, cx, cy, w, h, angle, tag, conf=-1, assign_id=-1, describe=''):
        """增加一个角度矩形对象"""
        one_dete_obj = DeteAngleObj(cx=cx, cy=cy, w=w, h=h, angle=angle, tag=tag, conf=conf, assign_id=assign_id, describe=describe)
        self._alarms.append(one_dete_obj)

    def add_obj_2(self, one_dete_obj):
        """增加一个检测框"""
        if isinstance(one_dete_obj, DeteObj) or isinstance(one_dete_obj, DeteAngleObj):
            one_dete_obj_new = copy.deepcopy(one_dete_obj)
            self._alarms.append(one_dete_obj_new)
        else:
            raise ValueError('one_dete_obj can only be DeteObj or DeteAngleObj')

    def draw_dete_res(self, save_path, assign_img=None, line_thickness=2, color_dict=None):
        """在图像上画出检测的结果"""
        #
        if color_dict is None:
            color_dict = {}
        #
        if not assign_img is None:
            img = assign_img
        else:
            if self.img is not None:
                img = np.array(self.img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif self.img_path:
                img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), 1)
            else:
                raise ValueError('need self.img or self.img_path')
        #
        for each_res in self._alarms:
            #
            if isinstance(each_res.tag, int) or isinstance(each_res, float):
                raise ValueError("tag should not be int or float")
            #
            if each_res.tag in color_dict:
                each_color = color_dict[each_res.tag]
            else:
                each_color = [random.randint(0, 255), random.randint(0,255), random.randint(0, 255)]
                color_dict[each_res.tag] = each_color

            tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))      # line thickness
            tf = max(tl - 2, 1)                                                 # font thickness
            #
            s_size = cv2.getTextSize(str('{:.0%}'.format(float(each_res.conf))), 0, fontScale=float(tl) / 3, thickness=tf)[0]
            t_size = cv2.getTextSize(each_res.tag, 0, fontScale=float(tl) / 3, thickness=tf)[0]

            if isinstance(each_res, DeteObj):
                c1, c2 =(each_res.x1, each_res.y1), (each_res.x2, each_res.y2)
                c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
                cv2.rectangle(img, (each_res.x1, each_res.y1), (each_res.x2, each_res.y2), color=each_color, thickness=tl)
                cv2.rectangle(img, c1, c2, each_color, -1)  # filled
                cv2.putText(img, '{}: {:.0%}'.format(str(each_res.tag), float(each_res.conf)), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)
            else:
                # 找到左上角的点
                pt_sorted_by_left = sorted(each_res.get_points(), key=lambda x:x[0])
                c1 = pt_sorted_by_left[0]
                c1 = (int(c1[0]), int(c1[1]))
                c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
                pts = np.array(each_res.get_points(), np.int)
                cv2.polylines(img, [pts], True, color=each_color, thickness=tl)
                #
                cv2.rectangle(img, c1, c2, each_color, -1)  # filled
                cv2.putText(img, '{}: {:.0%}'.format(str(each_res.tag), float(each_res.conf)), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

                # todo 得到小矩形的范围，再画上即可
                # cv2.fillPoly(img, [pts], color=[0,0,255])

        # 保存图片，解决保存中文乱码问题
        cv2.imencode('.jpg', img)[1].tofile(save_path)
        # return color_dict
        return img

    def do_func(self, assign_func):
        """对所有元素进行指定操作"""
        for each_dete_obj in self._alarms:
            assign_func(each_dete_obj)

    def do_nms(self, threshold=0.1, ignore_tag=False):
        """对结果做 nms 处理，"""
        # 参考：https://blog.csdn.net/shuzfan/article/details/52711706
        dete_res_list = copy.deepcopy(self._alarms)
        dete_res_list = sorted(dete_res_list, key=lambda x:x.conf, reverse=True)
        if len(dete_res_list) > 0:
            res = [dete_res_list.pop(0)]
        else:
            self._alarms = []
            return
        # 循环，直到 dete_res_list 中的数据被处理完
        while len(dete_res_list) > 0:
            each_res = dete_res_list.pop(0)
            is_add = True
            for each in res:
                # 计算每两个框之间的 iou，要是 nms 大于阈值，同时标签一致，去除置信度比较小的标签
                if ResTools.cal_iou(each, each_res, ignore_tag=ignore_tag) > threshold:
                    is_add = False
                    break
            # 如果判断需要添加到结果中
            if is_add is True:
                res.append(each_res)
        self._alarms = res

    def do_nms_center_point(self, ignore_tag=False):
        """中心点 nms，一个要素的中心点要是在另一个里面，去掉这个要素"""
        dete_obj_list = copy.deepcopy(self._alarms)
        dete_obj_list = sorted(dete_obj_list, key=lambda x:x.conf, reverse=True)
        if len(dete_obj_list) > 0:
            res = [dete_obj_list.pop(0)]
        else:
            self._alarms = []
            return
        # 循环，直到 dete_res_list 中的数据被处理完
        while len(dete_obj_list) > 0:
            each_res = dete_obj_list.pop(0)
            is_add = True
            for each in res:
                # fixme 这个逻辑存在一个问题，当 conf 高的中心点不在 conf 低的范围内，但是 conf 低的中心点在 conf 高的范围内，只会保留 conf 比较低的，如何才能只是保留
                if ResTools.point_in_poly(each_res.get_center_point(), each.get_points()) or ResTools.point_in_poly(each.get_center_point(), each_res.get_points()):
                    is_add = False
                    break
            # 如果判断需要添加到结果中
            if is_add is True:
                res.append(each_res)
        self._alarms = res

    def do_nms_in_assign_tags(self, tag_list, threshold=0.1):
        """在指定的 tags 之间进行 nms，其他类型的 tag 不受影响"""
        # 备份 alarms
        all_alarms = copy.deepcopy(self._alarms)
        # 拿到非指定 alarms
        self.filter_by_tags(remove_tag=tag_list)
        other_alarms = copy.deepcopy(self._alarms)
        # 拿到指定 alarms 进行 nms
        self.reset_alarms(all_alarms)
        self.filter_by_tags(need_tag=tag_list)
        self.do_nms(threshold, ignore_tag=True)
        # 添加其他类型
        for each_dete_obj in other_alarms:
            self._alarms.append(each_dete_obj)

    def update_tags(self, update_dict):
        """更新标签"""
        # tag 不在不更新字典中的就不进行更新
        for each_dete_res in self._alarms:
            if each_dete_res.tag in update_dict:
                each_dete_res.tag = update_dict[each_dete_res.tag]

    def reset_alarms(self, assign_alarms=None):
        """重置 alarms"""
        if assign_alarms is None:
            self._alarms = []
        else:
            self._alarms = assign_alarms

    # ------------------------------------------------ filter ----------------------------------------------------------

    def filter_by_area(self, area_th, update=True):
        """根据面积大小（像素个数）进行筛选, update 是否对 self 进行更新"""
        new_alarms, del_alarms = [], []
        for each_dete_tag in self._alarms:
            if each_dete_tag.get_area() >= area_th:
                new_alarms.append(each_dete_tag)
            else:
                del_alarms.append(each_dete_tag)
        if update:
            self._alarms = new_alarms
        return del_alarms

    def filter_by_area_ratio(self, ar=0.0006, update=True):
        """根据面积比例进行删选"""
        # get area
        th_area = float(self.width * self.height) * ar
        self.filter_by_area(area_th=th_area, update=update)

    def filter_by_tags(self, need_tag=None, remove_tag=None, update=True):
        """根据 tag 类型进行筛选"""
        new_alarms, del_alarms = [], []

        if (need_tag is not None and remove_tag is not None) or (need_tag is None and remove_tag is None):
            raise ValueError(" need tag and remove tag cant be None or not None in the same time")

        if isinstance(need_tag, str) or isinstance(remove_tag, str):
            raise ValueError("need list tuple or set not str")

        if need_tag is not None:
            need_tag = set(need_tag)
            for each_dete_tag in self._alarms:
                if each_dete_tag.tag in need_tag:
                    new_alarms.append(each_dete_tag)
                else:
                    del_alarms.append(each_dete_tag)
        else:
            remove_tag = set(remove_tag)
            for each_dete_tag in self._alarms:
                if each_dete_tag.tag not in remove_tag:
                    new_alarms.append(each_dete_tag)
                else:
                    del_alarms.append(each_dete_tag)
        if update:
            self._alarms = new_alarms
        return new_alarms

    def filter_by_conf(self, conf_th, assign_tag_list=None, update=True):
        """根据置信度进行筛选，指定标签就能对不同标签使用不同的置信度"""

        if not(isinstance(conf_th, int) or isinstance(conf_th, float)):
            raise ValueError("conf_th should be int or float")

        new_alarms, del_alarms = [], []
        for each_dete_obj in self._alarms:
            if assign_tag_list is not None:
                if each_dete_obj.tag not in assign_tag_list:
                    new_alarms.append(each_dete_obj)
                    continue
                else:
                    del_alarms.append(each_dete_obj)
            if each_dete_obj.conf >= conf_th:
                new_alarms.append(each_dete_obj)
            else:
                del_alarms.append(each_dete_obj)

        if update:
            self._alarms = new_alarms
        return new_alarms

    def filter_by_mask(self, mask, cover_index_th=0.5, need_in=True, update=True):
        """使用多边形 mask 进行过滤，mask 支持任意凸多边形，设定覆盖指数, mask 一连串的点连接起来的 [[x1,y1], [x2,y2], [x3,y3]], need_in is True, 保留里面的内容，否则保存外面的"""
        new_alarms, del_alarms = [], []
        for each_dete_obj in self._alarms:
            each_cover_index = ResTools.polygon_iou_1(each_dete_obj.get_points(), mask)
            # print("each_cover_index : ", each_cover_index)
            if each_cover_index > cover_index_th and need_in is True:
                new_alarms.append(each_dete_obj)
            elif each_cover_index < cover_index_th and need_in is False:
                new_alarms.append(each_dete_obj)
            else:
                del_alarms.append(each_dete_obj)
        if update:
            self._alarms = new_alarms
        return new_alarms

    def filter_by_dete_res_mask(self, mask_dete_res, cover_index_th=0.5, update=True):
        """将一个 deteRes 作为 mask 过滤 self"""
        dete_res_temp = DeteRes()
        for each_dete_obj in mask_dete_res:
            each_dete_res = self.deep_copy(copy_img=False)
            each_dete_res.filter_by_mask(each_dete_obj.get_points(), cover_index_th)
            dete_res_temp += each_dete_res
        if update:
            self.reset_alarms(dete_res_temp.alarms)
        return dete_res_temp

    def filter_by_func(self, func, update=True):
        """使用指定函数对 DeteObj 进行过滤"""
        new_alarms, del_alarms = [], []
        for each_dete_obj in self._alarms:
            if func(each_dete_obj):
                new_alarms.append(each_dete_obj)
            else:
                del_alarms.append(each_dete_obj)
        if update:
            self._alarms = new_alarms
        return new_alarms

    def filter_by_topn(self, nn, update=True):
        # 远景小目标过滤, 相对小的，从大到小排序，取前nn名的平均值/2作为阈值(籍天明，ljc)
        obj_area_list = []
        for dete_obj in self._alarms:
            obj_area_list.append(dete_obj.get_area())
        # find threshold
        threshold = -1
        obj_area_list.sort(reverse=True)
        if nn < len(obj_area_list):
            threshold = np.average(obj_area_list[:nn]) / 2
        # filter by area
        return self.filter_by_area(threshold, update=update)

    # ----------------------------------------------- set --------------------------------------------------------------

    def intersection(self, other):
        dete_res_tmp = self.deep_copy()
        dete_res_tmp.reset_alarms()
        #
        for each_dete_obj in self:
            if each_dete_obj in other:
                dete_res_tmp.add_obj_2(each_dete_obj)
        return dete_res_tmp

    def intersection_update(self, other):
        res = self.intersection(other)
        self.reset_alarms(res.alarms)

    def union(self, other):
        """就是加法操作"""
        dete_res_tmp = self.deep_copy()
        return dete_res_tmp + other

    def difference(self, other):
        """在 self 不在 other 中的"""
        diff_dete_res = self.deep_copy()
        for each_dete_obj in self:
            if each_dete_obj in other:
                diff_dete_res.del_dete_obj(each_dete_obj)
        return diff_dete_res

    def difference_update(self, other):
        res = self.difference(other)
        self.reset_alarms(res.alarms)

    def issubset(self, other):
        """是否为子集"""
        for each_dete_obj in self:
            if each_dete_obj not in other:
                return False
        return True

    def isupperset(self, other):
        """是否为超集"""
        for each_dete_obj in other:
            if each_dete_obj not in self:
                return False
        return True

    # ----------------------------------------------- del --------------------------------------------------------------

    def del_dete_obj(self, assign_dete_obj, del_all=False):
        """删除指定的一个 deteObj"""
        for each_dete_obj in copy.deepcopy(self._alarms):
            if each_dete_obj == assign_dete_obj:
                # del each_dete_obj # 使用 del 删除不了
                self._alarms.remove(each_dete_obj)
                # break or not
                if not del_all:
                    return

    # ----------------------------------------------- func -------------------------------------------------------------

    def get_dete_obj_list_by_func(self, func, is_deep_copy=False):
        """根据指定的方法获取需要的 dete_obj，可以指定是否执行深拷贝 """
        res = []
        for each_dete_obj in self._alarms:
            if func(each_dete_obj):
                if is_deep_copy:
                    res.append(each_dete_obj.deep_copy())
                else:
                    res.append(each_dete_obj)
        return res

    # ----------------------------------------------- set --------------------------------------------------------------

    def do_augment(self, augment_parameter, is_relative=True):
        """对检测框进行扩展"""

        # todo 这个函数不该存在，想办法融合到其他数据中

        for each_dete_obj in self._alarms:
            if isinstance(each_dete_obj, DeteObj):
                each_dete_obj.do_augment(augment_parameter=augment_parameter, width=self.width, height=self.height, is_relative=is_relative)
            # todo 使用的函数等待完善
            elif isinstance(each_dete_obj, DeteAngleObj):
                each_dete_obj.do_augment(augment_parameter=augment_parameter, width=self.width, height=self.height, is_relative=is_relative)

    # ----------------------------------------------- txkj -------------------------------------------------------------

    def get_fzc_format(self):
        """按照防振锤模型设定的输出格式进行格式化， [tag, index, int(x1), int(y1), int(x2), int(y2), str(score)], des"""
        res_list = []
        # 遍历得到多有的
        for each_obj in self._alarms:
            if isinstance(each_obj, DeteObj):
                res_list.append([each_obj.tag, each_obj.id, each_obj.x1, each_obj.y1, each_obj.x2, each_obj.y2, str(each_obj.conf), each_obj.des])
            elif isinstance(each_obj, DeteAngleObj):
                res_list.append([each_obj.tag, each_obj.id, each_obj.cx, each_obj.cy, each_obj.w, each_obj.h, each_obj.angle, each_obj.conf, each_obj.des])
        return res_list

    def print_as_fzc_format(self):
        """按照防振锤的格式打印出来"""
        for each in self.get_fzc_format():
            print(each)

    def get_result_construction(self, model_name="None", start_time=None, end_time=None):
        """返回规范的检测结果字典"""

        if not end_time:
            end_time = time.time()

        result = {
                  'filename': self.file_name,
                  'start_time': start_time,
                  'end_time': end_time,
                  'width': self.width,
                  'height': self.height,
                  'alarms': [],
                  'model_name':model_name,
                  }

        each_info = {}
        for each_dete_obj in self._alarms:
            each_info['position'] = [each_dete_obj.x1, each_dete_obj.y1, each_dete_obj.x2-each_dete_obj.x1, each_dete_obj.y2-each_dete_obj.y1]
            each_info['class'] = each_dete_obj.tag
            each_info['possibility'] = each_dete_obj.conf
            result['alarms'].append(copy.deepcopy(each_info))

        result['count'] = len(result['alarms'])
        return result

    def get_return_jsonify(self, script_name=None, obj_name=None):
        """获取返回信息"""
        # fixme 最好能强制全部使用并行模式，这样就省的麻烦了
        # 并行模式
        if obj_name:
            if len(self._alarms) > 0:
                return jsonify({script_name: {obj_name: self.save_to_json()}}), 200
            else:
                return jsonify({script_name: {obj_name: self.save_to_json()}}), 207
        # 非并行模式
        else:
            if len(self._alarms) > 0:
                return jsonify({script_name: self.save_to_json()}), 200
            else:
                return jsonify({script_name: self.save_to_json()}), 207

    # @DecoratorUtil.time_this
    def deep_copy(self, copy_img=False):
        """深拷贝，为了时间考虑，分享的是同一个 img 对象"""
        if copy_img:
            return copy.deepcopy(self)
        else:
            a = DeteRes()
            a.parse_auto = False
            a.height = self.height
            a.width = self.width
            a.xml_path = self.xml_path
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

    # ------------------------------------------------------------------------------------------------------------------

    def offset(self, x, y):
        """横纵坐标中的偏移量"""
        for each_dete_obj in self._alarms:
            each_dete_obj.do_offset(x, y)

    # @DecoratorUtil.time_this
    def crop_and_save(self, save_dir, augment_parameter=None, method=None, exclude_tag_list=None, split_by_tag=False, include_tag_list=None, assign_img_name=None, save_augment=False):
        """将指定的类型的结果进行保存，可以只保存指定的类型，命名使用标准化的名字 fine_name + tag + index, 可指定是否对结果进行重采样，或做特定的转换，只要传入转换函数
        * augment_parameter = [0.5, 0.5, 0.2, 0.2]
        * save_augment 是否保存为扩展后的范围，还是之前的范围
        """

        if len(self._alarms) == 0:
            # 要是没有可以剪切的要素就不浪费时间进行剪切了
            return

        if not self.img:
            raise ValueError ("need img_path or img")

        #
        if assign_img_name is not None:
            img_name = assign_img_name
        else:
            if self.img_path is not None :
                img_name = os.path.split(self.img_path)[1][:-4]
            else:
                raise ValueError("need self.img_path or assign_img_name")

        tag_count_dict = {}
        #
        for each_obj in self._alarms:
            # 只支持正框的裁切
            if not isinstance(each_obj, DeteObj):
                continue
            # 截图的区域
            bndbox = [each_obj.x1, each_obj.y1, each_obj.x2, each_obj.y2]
            # 排除掉不需要保存的 tag
            if include_tag_list is not None:
                if each_obj.tag not in include_tag_list:
                    continue

            if not exclude_tag_list is None:
                if each_obj.tag in exclude_tag_list:
                    continue

            # 计算这是当前 tag 的第几个图片
            if each_obj.tag not in tag_count_dict:
                tag_count_dict[each_obj.tag] = 0
            else:
                tag_count_dict[each_obj.tag] += 1
            # 图片扩展
            if augment_parameter is not None:
                bndbox = ResTools.region_augment(bndbox, [self.width, self.height], augment_parameter=augment_parameter)

            # 为了区分哪里是最新加上去的，使用特殊符号 -+- 用于标志
            if split_by_tag is True:
                each_save_dir = os.path.join(save_dir, each_obj.tag)
                if not os.path.exists(each_save_dir):
                    os.makedirs(each_save_dir)
            else:
                each_save_dir = save_dir

            # fixme 图像范围进行扩展，但是标注的范围不进行扩展，这边要注意
            if save_augment:
                each_name_str = each_obj.get_name_str(assign_loc=bndbox)
            else:
                each_name_str = each_obj.get_name_str()
            each_save_path = os.path.join(each_save_dir, '{0}-+-{1}.jpg'.format(img_name, each_name_str))

            try:
                # todo 对 bndbox 的范围进行检查
                each_crop = self.img.crop(bndbox)
                # 对截图的图片自定义操作, 可以指定缩放大小之类的
                if method is not None:
                    each_crop = method(each_crop)
                # 保存截图
                # each_crop.save(each_save_path, quality=95)
                each_crop.save(each_save_path)
            except Exception as e:
                # 当遇到错误的图片会报错，
                # todo 图片损坏使用 cv2.crop 截取图片不会报错，是否考虑使用 cv2 的截图获取部分破损的图片
                print(e)

    def crop_and_save_new(self, save_dir, augment_parameter=None, method=None, exclude_tag_list=None, split_by_tag=False, include_tag_list=None, assign_img_name=None, save_augment=False):
        """将指定的类型的结果进行保存，可以只保存指定的类型，命名使用标准化的名字 fine_name + tag + index, 可指定是否对结果进行重采样，或做特定的转换，只要传入转换函数
        * augment_parameter = [0.5, 0.5, 0.2, 0.2]
        * save_augment 是否保存为扩展后的范围，还是之前的范围
        """

        if self.img_ndarry is None:
            self.img_ndarry = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), 1)
        #
        if assign_img_name is not None:
            img_name = assign_img_name
        else:
            if self.img_path is not None :
                img_name = os.path.split(self.img_path)[1][:-4]
            else:
                raise ValueError("need self.img_path or assign_img_name")

        tag_count_dict = {}
        #
        for each_obj in self._alarms:
            # 只支持正框的裁切
            if not isinstance(each_obj, DeteObj):
                continue
            # 截图的区域
            bndbox = [each_obj.x1, each_obj.y1, each_obj.x2, each_obj.y2]
            # 排除掉不需要保存的 tag
            if include_tag_list is not None:
                if each_obj.tag not in include_tag_list:
                    continue

            if not exclude_tag_list is None:
                if each_obj.tag in exclude_tag_list:
                    continue

            # 计算这是当前 tag 的第几个图片
            if each_obj.tag not in tag_count_dict:
                tag_count_dict[each_obj.tag] = 0
            else:
                tag_count_dict[each_obj.tag] += 1
            # 图片扩展
            if augment_parameter is not None:
                bndbox = ResTools.region_augment(bndbox, [self.width, self.height], augment_parameter=augment_parameter)

            # 为了区分哪里是最新加上去的，使用特殊符号 -+- 用于标志
            if split_by_tag is True:
                each_save_dir = os.path.join(save_dir, each_obj.tag)
                if not os.path.exists(each_save_dir):
                    os.makedirs(each_save_dir)
            else:
                each_save_dir = save_dir

            # fixme 图像范围进行扩展，但是标注的范围不进行扩展，这边要注意
            if save_augment:
                each_name_str = each_obj.get_name_str(assign_loc=bndbox)
            else:
                each_name_str = each_obj.get_name_str()
            each_save_path = os.path.join(each_save_dir, '{0}-+-{1}.jpg'.format(img_name, each_name_str))

            try:
                # todo 对 bndbox 的范围进行检查
                each_crop = self.img.crop(bndbox)
                # 对截图的图片自定义操作, 可以指定缩放大小之类的
                if method is not None:
                    each_crop = method(each_crop)
                # 保存截图
                # each_crop.save(each_save_path, quality=95)
                each_crop.save(each_save_path)
            except Exception as e:
                # 当遇到错误的图片会报错，
                # todo 图片损坏使用 cv2.crop 截取图片不会报错，是否考虑使用 cv2 的截图获取部分破损的图片
                print(e)

    def crop_angle_and_save(self, save_dir, augment_parameter=None, method=None, exclude_tag_list=None, split_by_tag=False):
        """将指定的类型的结果进行保存，可以只保存指定的类型，命名使用标准化的名字 fine_name + tag + index, 可指定是否对结果进行重采样，或做特定的转换，只要传入转换函数
        * augment_parameter = [0.2, 0.2] w,h的扩展比例
        """
        img_name = os.path.split(self.img_path)[1][:-4]
        tag_count_dict = {}
        #
        for each_obj in self._alarms:
            # 去除正框
            if not isinstance(each_obj, DeteAngleObj): continue
            # 排除掉不需要保存的 tag
            if not exclude_tag_list is None:
                if each_obj.tag in exclude_tag_list:
                    continue
            # 计算这是当前 tag 的第几个图片
            if each_obj.tag not in tag_count_dict:
                tag_count_dict[each_obj.tag] = 0
            else:
                tag_count_dict[each_obj.tag] += 1
            # 图片扩展
            loc_str = "[{0}_{1}_{2}_{3}_{4}]".format(each_obj.cx, each_obj.cy, each_obj.w, each_obj.h, each_obj.angle)

            # 为了区分哪里是最新加上去的，使用特殊符号 -+- 用于标志
            if split_by_tag is True:
                each_save_dir = os.path.join(save_dir, each_obj.tag)
                if not os.path.exists(each_save_dir): os.makedirs(each_save_dir)
            else:
                each_save_dir = save_dir

            each_name_str = each_obj.get_name_str()
            each_save_path = os.path.join(each_save_dir, '{0}-+-{1}.jpg'.format(img_name, each_name_str))
            cx, cy, w, h, angle = each_obj.cx, each_obj.cy, each_obj.w, each_obj.h, each_obj.angle
            # 范围扩展
            if augment_parameter is not None:
                w += w * augment_parameter[0]
                h += h * augment_parameter[1]
            # 裁剪
            each_crop = ResTools.crop_angle_rect(self.get_img_array(), ((cx, cy), (w, h), angle))
            if method is not None: each_crop = method(each_crop)
            # crop = Image.fromarray(each_crop)
            # crop.save(each_save_path)

            cv2.imencode('.jpg', each_crop)[1].tofile(each_save_path)

    def crop_with_xml(self, augment_parameter, save_dir, split_by_tag=False, need_tags=None):
        """保存裁剪结果，结果带着 xml"""
        #
        for each_dete_obj in self._alarms:
            if need_tags:
                if each_dete_obj.tag not in need_tags:
                    continue

            x_min, y_min, x_max, y_max = each_dete_obj.get_rectangle()
            new_x_min, new_y_min, new_x_max, new_y_max = ResTools.region_augment(each_dete_obj.get_rectangle(), (self.width, self.height), augment_parameter=augment_parameter)
            # todo 获取相对位置，即为 xml 中的位置值
            x1 = x_min - new_x_min + 1
            x2 = x_max - x_min + x1
            y1 = y_min - new_y_min + 1
            y2 = y_max - y_min + y1

            a = DeteRes()
            a.add_obj(x1=x1, y1=y1, x2=x2, y2=y2, tag=each_dete_obj.tag, conf=each_dete_obj.conf, assign_id=0)
            each_name = FileOperationUtil.bang_path(self.xml_path)[1]
            if split_by_tag:
                each_save_dir = os.path.join(save_dir, each_dete_obj.tag)
                os.makedirs(each_save_dir, exist_ok=True)
                each_xml_path = os.path.join(each_save_dir, each_name + "-+-" +each_dete_obj.get_name_str([new_x_min, new_y_min, new_x_max, new_y_max])+'.xml')
                each_img_path = os.path.join(each_save_dir, each_name + "-+-" +each_dete_obj.get_name_str([new_x_min, new_y_min, new_x_max, new_y_max])+'.jpg')
            else:
                each_xml_path = os.path.join(save_dir, each_name + "-+-" + each_dete_obj.get_name_str([new_x_min, new_y_min, new_x_max, new_y_max])+'.xml')
                each_img_path = os.path.join(save_dir, each_name + "-+-" + each_dete_obj.get_name_str([new_x_min, new_y_min, new_x_max, new_y_max])+'.jpg')

            a.save_to_xml(each_xml_path)
            each_img = self.img.crop([new_x_min, new_y_min, new_x_max, new_y_max])
            each_img.save(each_img_path)

    # ------------------------------------------------------------------------------------------------------------------

    def has_tag(self, assign_tag):
        """是否存在指定的标签"""
        for each_dete_obj in self._alarms:
            if each_dete_obj.tag == assign_tag:
                return True
        return False

    def save_assign_range(self, assign_range, save_dir, save_name=None, iou_1=0.85):
        """保存指定范围，同时保存图片和 xml """

        x1, y1, x2, y2 = int(assign_range[0]),int(assign_range[1]),int(assign_range[2]),int(assign_range[3])
        assign_dete_obj = DeteObj(x1=x1, y1=y1, x2=x2, y2=y2, tag='None', conf=-1)

        offset_x, offset_y = -int(assign_range[0]), -int(assign_range[1])
        height, width = y2 - y1, x2 - x1

        new_alarms = []
        # 这边要是直接使用 .copy 的话，alarms 里面的内容还是会被改变的, list 的 .copy() 属于 shallow copy 是浅复制，对浅复制中的可变类型修改的时候原数据会受到影响，https://blog.csdn.net/u011995719/article/details/82911392
        for each_dete_obj in copy.deepcopy(self._alarms):
            # 计算重合度
            each_iou_1 = ResTools.cal_iou_1(each_dete_obj, assign_dete_obj, ignore_tag=True)
            if each_iou_1 > iou_1:
                # 对结果 xml 的范围进行调整
                each_dete_obj.do_offset(offset_x, offset_y)
                # 支持斜框和正框
                if isinstance(each_dete_obj, DeteAngleObj):
                    each_dete_obj_new = each_dete_obj.get_dete_obj().deep_copy()
                elif isinstance(each_dete_obj, DeteObj):
                    each_dete_obj_new = each_dete_obj.deep_copy()
                else:
                    raise ValueError("obj type in alrms error")

                # 修正目标的范围
                if each_dete_obj_new.x1 < 0:
                    each_dete_obj_new.x1 = 0
                if each_dete_obj_new.y1 < 0:
                    each_dete_obj_new.y1 = 0
                if each_dete_obj_new.x2 > width:
                    each_dete_obj_new.x2 = width
                if each_dete_obj_new.y2 > height:
                    each_dete_obj_new.y2 = height
                #
                new_alarms.append(each_dete_obj_new)

        # 保存 xml
        if save_name is None:
            loc_str = "[{0}_{1}_{2}_{3}]".format(assign_range[0], assign_range[1], assign_range[2], assign_range[3])
            save_name = os.path.split(self.xml_path)[1].strip('.xml')+ '-+-' + loc_str

        xml_save_dir = os.path.join(save_dir, 'Annotations')
        img_save_dir = os.path.join(save_dir, 'JPEGImages')
        xml_save_path = os.path.join(xml_save_dir, save_name + '.xml')
        jpg_save_path = os.path.join(img_save_dir, save_name + '.jpg')
        os.makedirs(xml_save_dir, exist_ok=True)
        os.makedirs(img_save_dir, exist_ok=True)
        #
        self.save_to_xml(xml_save_path, new_alarms)

        # # 保存 jpg
        crop = self.img.crop(assign_range)
        crop.save(jpg_save_path, quality=95)

    def count_tags(self):
        """统计标签数"""
        tags_count = {}
        for each_dete_res in self._alarms:
            each_tag = each_dete_res.tag
            if each_tag in tags_count:
                tags_count[each_tag] += 1
            else:
                tags_count[each_tag] = 1
        return tags_count

    def angle_obj_to_obj(self):
        """将斜框全部转为正框"""
        new_alarms = []
        for each_obj in self._alarms:
            if isinstance(each_obj, DeteObj):
                new_alarms.append(each_obj)
            elif isinstance(each_obj, DeteAngleObj):
                each_obj = each_obj.get_dete_obj()
                new_alarms.append(each_obj)
        self._alarms = new_alarms

