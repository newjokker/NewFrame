import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

import sklearn.cluster as skc  # 密度聚类
from math import sqrt, pow


def decode_seg_map_sequence(label_masks, dataset='invoice',class_num = 2):  # 数据集名称
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset,class_num)
        rgb_masks.append(rgb_mask)
    #rgb_masks = np.array(rgb_masks).transpose([0, 3, 2, 1])
    return rgb_masks

def get_label_colours(class_num):
    if class_num == 2:
        return np.array([[0, 0, 0], [255, 255, 255]])
    else:
        return np.random.randint(0,256,(class_num,3))

def decode_segmap(label_mask, dataset, n_classes,plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    label_colours = get_label_colours(n_classes)
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r  # /255 未归一化处理；由于只有一类目标，故将g、b通道像素值设为0避免重复后续输出点的坐标 +++++++++++++++++++++
    rgb[:, :, 1] = g#np.zeros(shape=r.shape)  # g
    rgb[:, :, 2] = b#np.zeros(shape=r.shape)  # b
    #zero_image = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))

    return rgb
    #k = np.ones((3, 3), np.uint8)  # 对mask图进行开运算，用于去除毛边和瑕疵点，k越大去除越多
    #rgb = cv2.morphologyEx(rgb, cv2.MORPH_OPEN, k)

def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_invoice_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_invoice_labels():  # 设置背景及目标mask颜色 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return np.array([[0, 0, 0], [255, 255, 255]])


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


#  Douglas-Peuker 抽稀算法 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def point2LineDistance(point_a, point_b, point_c):
    # 首先计算b c 所在直线的斜率和截距
    if point_b[0] == point_c[0]:
        return 9999999
    slope = (point_b[1] - point_c[1]) / (point_b[0] - point_c[0])
    intercept = point_b[1] - slope * point_b[0]

    # 计算点a到b c所在直线的距离
    distance = abs(slope * point_a[0] - point_a[1] + intercept) / sqrt(1 + pow(slope, 2))
    return distance


class DouglasPeuker(object):
    def __init__(self):
        self.threshold = 10  # 抽稀阈值，推荐值10—15.——————————————————————————————————————————————————————————————
        self.qualify_list = list()
        self.disqualify_list = list()

    def diluting(self, point_list):
        """
        抽稀
        :param point_list:二维点列表
        :return:
        """
        if len(point_list) < 3:
            self.qualify_list.extend(point_list[::-1])
        else:
            # 找到与收尾两点连线距离最大的点
            max_distance_index, max_distance = 0, 0
            for index, point in enumerate(point_list):
                if index in [0, len(point_list) - 1]:
                    continue
                distance = point2LineDistance(point, point_list[0], point_list[-1])
                if distance > max_distance:
                    max_distance_index = index
                    max_distance = distance

            # 若最大距离小于阈值，则去掉所有中间点。 反之，则将曲线按最大距离点分割
            if max_distance < self.threshold:
                self.qualify_list.append(point_list[-1])
                self.qualify_list.append(point_list[0])
            else:
                # 将曲线按最大距离的点分割成两段
                sequence_a = point_list[:max_distance_index]
                sequence_b = point_list[max_distance_index:]

                for sequence in [sequence_a, sequence_b]:
                    if len(sequence) < 3 and sequence == sequence_b:
                        self.qualify_list.extend(sequence[::-1])
                    else:
                        self.disqualify_list.append(sequence)

    def main(self, point_list):
        self.diluting(point_list)
        while len(self.disqualify_list) > 0:
            self.diluting(self.disqualify_list.pop())

        print("抽稀阈值为： ")
        print(self.threshold)
        print("原始点的数量为： ")
        print(len(point_list))  # 原始点的数量
        print("简化后点的数量为： ")
        print(len(self.qualify_list))  # 简化后的点的数量
        print("简化后点的坐标为： ")
        return self.qualify_list#)  # 简化后的点




