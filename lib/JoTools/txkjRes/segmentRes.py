# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import cv2
import copy
from labelme import utils
from ..utils.JsonUtil import JsonUtil
from ..utils.FileOperationUtil import FileOperationUtil
from .segmentObj import SegmentObj
import numpy as np
from skimage.measure import find_contours
from PIL import Image

# todo 指定输出 mask 的格式，如何将每个 mask obj 对应到 label 上去, 因为寻找外轮廓的时候 是从值小点的到值大的这么个顺序找的，所以只要记录下
# todo mask 试一下不要写为 bool 类型的，而是写为 int 类型的，最多 255 对象，每个对象一个值，这样说不定找边缘的时候就能分开了

# todo 好像对象过多的时候 labelme 上无法展示，看看到底是什么原因


class SegmentRes(object):

    def __init__(self, json_path=None):

        self.version = "3.16.1"
        self.image_width = ""
        self.image_height = ""
        self.shapes = []
        self.img_path = ""
        self.line_color = [0,255,0,128]
        self.fill_color = [255,0,0,128]
        self.image_data = None                  # 转为 array 的 bs64 字符串
        self.img = None                         # PIL.Image 对象
        self.image_data_bs64 = None             # 保存在 json 文件中的 bs64 字符串
        self.flags = ""
        self.json_path = json_path
        self.mask = None

    def __getitem__(self, index):
        """按照 index 取对应的对象"""
        return self.shapes[index]

    def __len__(self):
        """返回要素的个数"""
        return len(self.shapes)

    def parse_json_info(self, json_path=None, parse_img=False, parse_mask=False):
        """解析 json 的信息, 可以选择是否解析 img 和 mask"""

        # todo 读取的时候，每一个的 label 是不一样的 从 test1 到 testn 出现一个一样的 segmentObj 增加一个序号

        self.json_path = json_path

        if json_path:
            a = JsonUtil.load_data_from_json_file(json_path, encoding='GBK')
        else:
            a = JsonUtil.load_data_from_json_file(self.json_path, encoding='GBK')

        # if a is None:
        #     return

        # parse attr
        self.version = a["version"] if "version" in a else ""
        self.image_width = a["imageWidth"] if "imageWidth" in a else ""
        self.image_height = a["imageHeight"] if "imageWidth" in a else ""
        self.img_path = a["imagePath"] if "imagePath" in a else ""
        self.line_color = a["lineColor"] if "lineColor" in a else []
        self.fill_color = a["fillColor"] if "fillColor" in a else []
        self.image_data_bs64 = a["imageData"]

        # 需要拿到 img 才知道图像的大小，属性中的图像大小可能出问题
        if parse_img or parse_mask:
            self.image_data = utils.img_b64_to_arr(a["imageData"]) if "imageData" in a else ""
        self.flags = a["flags"] if "flags" in a else ""

        # parse shape
        label_name_dict = {}
        lables_dict = {}
        value_index = 1

        # todo 下面的代码要精简一下，很多地方用不到
        for each_shape in a["shapes"]:
            each_label = each_shape["label"] if "label" in each_shape else ""
            # strip number
            each_label_no_number = each_label.strip("0123456789")
            # fixme label_no_number 是 Jo12 应该是 Jo， yo2 应该是 yo
            if each_label_no_number not in lables_dict:
                lables_dict[each_label_no_number] = value_index
                value_index += 1
            label_name_dict[each_label] = lables_dict[each_label_no_number]
            #
            each_shape_points = each_shape["points"] if "points" in each_shape else []
            each_type = each_shape["shape_type"] if "shape_type" in each_shape else ""
            each_obj = SegmentObj(label=each_label, points=each_shape_points, shape_type=each_type, mask_value=each_label_no_number)
            self.shapes.append(each_obj)

        # parse mask
        if parse_mask:
            # fixme mask 的 channel 必须和 box 的个数一样多，否则会报错【好像还不一定，好像是问题】
            _, self.mask = utils.shapes_to_label(self.image_data.shape, a["shapes"], label_name_dict)

    def save_to_josn(self, json_path):
        """保存为json数据格式"""

        json_info = {"version":"", "imageWidth":"", "imageHeight":"", "imagePath":"", "lineColor":"", "fillColor":"", "imageData":"", "shapes":[]}

        if self.version:
            json_info["version"] = self.version
        if self.image_width:
            json_info["imageWidth"] = self.image_width
        if self.image_height:
            json_info["imageHeight"] = self.image_height
        if self.img_path:
            json_info["imagePath"] = self.img_path
        if self.line_color:
            json_info["lineColor"] = self.line_color
        if self.fill_color:
            json_info["fillColor"] = self.fill_color
        # --------------------------------------------------------------------------
        for each_shape in self.shapes:
            each_shape_info = {
                "label": each_shape.label,
                "points": each_shape.points,
                "shape_type": each_shape.shape_type,
                "line_color": each_shape.line_color,
                "fill_color": each_shape.fill_color}
            json_info["shapes"].append(each_shape_info)
        #
        if self.image_data_bs64:
            json_info["imageData"] = self.image_data_bs64
        elif self.img_path:
            img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), 1)
            self.image_data_bs64 = utils.img_arr_to_b64(img).decode('utf-8')
            json_info["imageData"] = self.image_data_bs64
        else:
            raise ValueError("self.image_data_bs64 and img_path can not empty both")
        # save
        JsonUtil.save_data_to_json_file(json_info, json_path, encoding="GBK")

    def get_segment_obj_from_mask(self, mask, each_mask_point_numb=5, json_label_dict=None):
        """从掩膜中提取关键点, each_mask_point_numb 指定每个 mask 大概用多少点进行描绘, json_label_dict 指定每个值对应的 label 的值"""

        # mask is path str
        if isinstance(mask, str):
            mask = cv2.imdecode(np.fromfile(mask, dtype=np.uint8), 1)
            if mask. ndim >= 3:
                mask = mask[:,:,0]
            elif mask.ndim == 2:
                pass
            else:
                raise TypeError("mask's dim should be >= 2")

        # 读取图像的长宽，如果之前没有读取的话
        if (not self.image_width) or (not self.image_height):
            self.image_height, self.image_width = mask.shape[:2]

        # 找到轮廓点
        contours = find_contours(mask, 0.5)
        label_index = 1
        for contour in contours:
            # filter small
            if len(contour) <= each_mask_point_numb:
                continue
            #
            del_list = [i for i in range(1, len(contour) - 1) if i % int(len(contour) / each_mask_point_numb) != 0]
            contour = np.delete(contour, del_list, axis=0)
            # points_list.append(contour)
            each_points_list = []
            for each_point in contour:
                each_points_list.append([each_point[1], each_point[0]])
            # fixme 完善一下增加 label 名
            each_segment_obj = SegmentObj(label="test{0}".format(label_index), points=each_points_list, shape_type="polygon", mask=None, mask_value=None)
            self.shapes.append(each_segment_obj)
            label_index += 1

    def crop_and_save(self, save_dir, assign_name=""):
        """找到每个对象的矩形范围，进行裁剪"""

        # 将图像中 mask 对应的部分改为不同的颜色

        if (self.image_data is not None) and (not self.img):
            self.img = Image.fromarray(self.image_data)

        if os.path.exists(self.img_path) and (not self.img):
            self.img = Image.open(self.img_path)

        if not self.img:
            raise ValueError("self.img_path self.img self.img_data all empty")

        img_name = FileOperationUtil.bang_path(self.json_path)[1]

        for obj_index, each_segment_obj in enumerate(self.shapes):
            each_rect = each_segment_obj.get_rectangle()
            each_crop = self.img.crop(each_rect)
            try:
                each_crop.save(os.path.join(save_dir, "{0}_{1}_{2}.jpg".format(obj_index, img_name, each_segment_obj.label)))
            except Exception as e:
                print(e)

    def filter_segment_obj_by_lables(self, include_labels=None, exclude_labels=None):
        """对 shape 中的 segment obj 对象进行过滤 """

        new_shapes = []

        if include_labels:
            for each_segment_obj in self.shapes:
                if each_segment_obj.label in include_labels:
                    new_shapes.append(each_segment_obj)
        elif exclude_labels:
            for each_segment_obj in self.shapes:
                if each_segment_obj.label not in exclude_labels:
                    new_shapes.append(each_segment_obj)
        else:
            return

        self.shapes = new_shapes

    def save_mask(self, save_path=None):
        """将 mask 保存为图片文件"""
        # fixme 最好不要保存，保存后的结果比较大，还不如临时读取
        np.save(save_path, self.mask.astype(np.bool))

    def print_as_fzc_format(self):
        """按照防振锤的格式进行打印"""
        for each_shape in self.shapes:
            print(each_shape.get_format_list())

    def draw_points(self):
        """在图上画出点"""

        if (self.image_data is not None) and (not self.img):
            self.img = Image.fromarray(self.image_data)

        if os.path.exists(self.img_path) and (not self.img):
            self.img = Image.open(self.img_path)

        if not self.img:
            raise ValueError("self.img_path self.img self.img_data all empty")


        




if __name__ == "__main__":

    img_dir = r"C:\Users\14271\Desktop\mask_test_res"
    mask_dir = r"C:\Users\14271\Desktop\mask_test_res"
    save_dir = r"C:\Users\14271\Desktop\mask_test_res\json"

    for each_img_path in FileOperationUtil.re_all_file(img_dir, endswitch=['.jpg']):
        each_mask_path = os.path.join(mask_dir, FileOperationUtil.bang_path(each_img_path)[1] + '_mask.png')
        each_save_path = os.path.join(save_dir, FileOperationUtil.bang_path(each_img_path)[1] + '.json')

        a = SegmentRes()
        a.img_path = each_img_path
        a.get_segment_obj_from_mask(each_mask_path, each_mask_point_numb=30)
        a.save_to_josn(each_save_path)













