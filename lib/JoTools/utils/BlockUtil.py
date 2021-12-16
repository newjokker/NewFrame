# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL.Image import Image
import time
import random
from math import ceil


class BlockUtil():

    def __init__(self, width, height, block_x, block_y, mode=0):
        self.width = width
        self.height = height
        self.block_x = block_x
        self.block_y = block_y
        self.block_width = None
        self.block_height = None
        self.mode = mode
        #
        if width/float(self.block_x) < 1 or height/float(self.block_y) < 1:
            raise ValueError("width or height is too small compare to self.block_x, self.block_y")

        # 设置 block 长宽为浮点数，解决为整数时候遇到的诸多问题，block 的范围在使用的时候再去计算，
        self.block_width = width/self.block_x
        self.block_height = height/self.block_y

    def get_block_range(self, block_x, block_y, do_augment=None):
        """获取某一个 block 的范围"""
        if self.mode == 0:
            x1 = int(self.block_width * block_x)
            y1 = int(self.block_height * block_y)
            x2 = ceil(self.block_width * (block_x + 1))
            y2 = ceil(self.block_height * (block_y + 1))
            # do augment
            if do_augment:
                x1, y1, x2, y2 = self.do_augment((x1, y1, x2, y2), do_augment, is_relative=False)
            # change value range
            x1 = max(0, x1)
            y1 = max(0, y1)
            y2 = min(self.height-1, y2)
            x2 = min(self.width-1, x2)
            #
            return (x1, y1, x2, y2)

    def get_block_heigh_and_width(self, block_x, block_y):
        x1, y1, x2, y2 = self.get_block_range(block_x, block_y)
        return (x2-x1, y2-y1)

    @staticmethod
    def do_augment(region_range, augment_parameter, is_relative=False):
        """对框进行扩展，这边传入的绝对比例，或者相对"""
        (x1, y1, x2, y2) = region_range
        region_width = int(x2 - x1)
        region_height = int(y2 - y1)
        #
        if is_relative:
            new_x_min = x1 - int(region_width * augment_parameter[0])
            new_x_max = x2 + int(region_width * augment_parameter[1])
            new_y_min = y1 - int(region_height * augment_parameter[2])
            new_y_max = y2 + int(region_height * augment_parameter[3])
        else:
            new_x_min = x1 - int(augment_parameter[0])
            new_x_max = x2 + int(augment_parameter[1])
            new_y_min = y1 - int(augment_parameter[2])
            new_y_max = y2 + int(augment_parameter[3])
        return (new_x_min, new_y_min, new_x_max, new_y_max)

    def draw_blocks(self, block_index_list):
        bg = np.array(np.zeros([self.height, self.width, 3]), dtype=np.int8)
        for each_block in block_index_list:
            print('-'*30)
            print(a.get_block_heigh_and_width(*each_block))
            x1, y1, x2, y2 = self.get_block_range(each_block[0], each_block[1], do_augment=[5,5,5,5])
            # x1, y1, x2, y2 = self.get_block_range(each_block[0], each_block[1])
            each_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            cv2.rectangle(bg, (x1, y1), (x2, y2), color=each_color, thickness=1)

        plt.imshow(bg)
        plt.show()


if __name__ == "__main__":

    assign_block_x = 13
    assign_block_y = 15

    a = BlockUtil(200, 200, assign_block_x, assign_block_y)

    label_list = [(x, y) for x in range(assign_block_x) for y in range(assign_block_y)]

    # a.draw_blocks([(1,1), (3,2), (4,1), (2,3), (4,4)])
    a.draw_blocks(label_list)







































