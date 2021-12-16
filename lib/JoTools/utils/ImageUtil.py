# -*- coding: utf-8  -*-
# -*- author: jokker -*-

# todo 调整函数名，让函数名更加有自解释性

import numpy as np
import random
from matplotlib.pylab import plt
from PIL import Image, ImageDraw, ImageFont
import imageio
from PIL import Image
import os
# import h5py


# todo 转为黑白图片


class ImageUtil(object):

    def __init__(self, image_path=None):
        self.__img_path = image_path

        # get image_mat from file
        if self.__img_path:
            self.__img_mat = self.get_img_mat_from_file(self.__img_path)
        else:
            self.__img_mat = None

    def add_border_line(self, line_weight=1, line_color=(1, 1, 1), line_inside=True):
        """ add line in the border of self, line_inside = True --> not change the shape of self.img_mat"""
        row_num, col_num = self.__img_mat.shape[:2]
        #
        if not line_inside:
            image_mat_new = np.zeros(
                (self.__img_mat.shape[0] + line_weight * 2, self.__img_mat.shape[1] + line_weight * 2, 4),
                dtype=np.uint8)
            image_mat_new[line_weight:row_num + line_weight, line_weight:col_num + line_weight, :] = self.__img_mat
            self.__img_mat = image_mat_new
        #
        row_num, col_num = self.__img_mat.shape[:2]
        mask = np.zeros(self.__img_mat.shape[:2], dtype=np.bool)
        mask[:line_weight, :] = True  # upper
        mask[:, :line_weight] = True  # left
        mask[row_num - line_weight:row_num, :] = True  # bottom
        mask[:, col_num - line_weight:col_num] = True  # right
        #
        self.__img_mat[mask, 0] = line_color[0]
        self.__img_mat[mask, 1] = line_color[1]
        self.__img_mat[mask, 2] = line_color[2]
        self.__img_mat[mask, 3] = 255

    def cat(self, cat_img, direction=0):
        """cat with another ImageUtil ==> ImageUtil, direction --> 0,1,2,3 --> left, right, bottom, up"""
        # todo Support for new functions: do not change cat_img's shape

        row_num, col_num = cat_img.get_img_mat().shape[:2]
        # find what shape cat_img should be
        if direction in [0, 1, '0', '1']:
            assign_row_num = self.__img_mat.shape[0]
            assign_col_num = int((float(col_num) / row_num) * assign_row_num)
        else:
            assign_col_num = self.__img_mat.shape[1]
            assign_row_num = int((float(row_num) / col_num) * assign_col_num)
        # change img_mat's shape
        cat_img.convert_to_assign_shape((assign_col_num, assign_row_num))
        image_mat_b = cat_img.get_img_mat()

        if direction in [0, '0']:
            new_image_mat = np.hstack((image_mat_b, self.__img_mat))
        elif direction in [1, '1']:
            new_image_mat = np.hstack((self.__img_mat, image_mat_b))
        elif direction in [2, '2']:
            new_image_mat = np.vstack((self.__img_mat, image_mat_b))
        elif direction in [3, '3']:
            new_image_mat = np.vstack((image_mat_b, self.__img_mat))
        else:
            raise TypeError('direction can only in [0, 1, 2, 3, "0", "1", "2", "3"]')
        self.__img_mat = new_image_mat

    def draw(self, ele_to_draw, assign_loc=(0, 0), assign_angle=0):
        """
        shape_b贴在shape_a上，只要指定一个左上角的位置即可,超出的部分可以自己增长，设置默认的无效值即可
        :param ele_to_draw:
        :param assign_loc: (row_index, column_index);(高向下移动，宽向左移动)
        :param assign_angle: 指定相对位置的叫，0：被粘贴图到粘贴图左上角的相对距离，1:右上角，2：左下角，3：右下角
        :return: image_mat
        """
        # FIXME 支持两种模式，替换模式和粘贴模式，一种不需要计算透明度的叠加，一种需要计算透明度的叠加

        image_mat_a = self.get_img_mat().copy()
        image_mat_b = ele_to_draw.get_img_mat().copy()

        # 使用矩阵镜像，可以操作四个角
        if assign_angle in [0, '0']:
            # 左上角
            pass
        elif assign_angle in [1, '1']:
            # 右上角
            image_mat_a = np.fliplr(image_mat_a)
            image_mat_b = np.fliplr(image_mat_b)
        elif assign_angle in [2, '2']:
            # 左下角
            image_mat_a = np.flipud(image_mat_a)
            image_mat_b = np.flipud(image_mat_b)
        elif assign_angle in [3, '3']:
            # 右下角
            image_mat_a = np.fliplr(np.flipud(image_mat_a))
            image_mat_b = np.fliplr(np.flipud(image_mat_b))
        else:
            raise TypeError('assign angle can only in [0,1,2,3, "0","1","2","3"]')

        # 左上角坐标
        x, y = assign_loc[0], assign_loc[1]
        # 贴图
        row_num_b, col_num_b = image_mat_b.shape[:2]  # 获得被贴图片的长宽
        row_num_a, col_num_a = image_mat_a.shape[:2]  # 获得画布的长宽
        # 判断画布是否不够
        if x >= row_num_a or y >= col_num_a:
            raise TypeError('贴图位置超出画布范围')
        elif x + row_num_b > row_num_a or y + col_num_b > col_num_a:
            y_max = min(y + col_num_b, col_num_a)
            x_max = min(x + row_num_b, row_num_a)
            image_mat_a[x:x_max, y:y_max, :] = ImageUtil.__get_com_color(image_mat_a[x:x_max, y:y_max, :],
                                                                         image_mat_b[:x_max - x, :y_max - y, :])
        else:
            image_mat_a[x:x + row_num_b, y:y + col_num_b, :] = ImageUtil.__get_com_color(
                image_mat_a[x:x + row_num_b, y:y + col_num_b, :], image_mat_b)

        # 抵消之前的矩阵操作
        if assign_angle in [0, '0']:
            # 左上角
            pass
        elif assign_angle in [1, '1']:
            # 右上角
            image_mat_a = np.fliplr(image_mat_a)
        elif assign_angle in [2, '2']:
            # 左下角
            image_mat_a = np.flipud(image_mat_a)
        elif assign_angle in [3, '3']:
            # 右下角
            image_mat_a = np.fliplr(np.flipud(image_mat_a))

        self.__img_mat = image_mat_a

    def create_img_mat(self, row_num, col_num, fill_color=(255, 255, 255)):
        """create a blank img_mat,"""
        # create a blank
        self.__img_mat = np.ones((row_num, col_num, 4), dtype=np.uint8) * 255
        # set color
        self.__img_mat[:, :, 0] = fill_color[0]
        self.__img_mat[:, :, 1] = fill_color[1]
        self.__img_mat[:, :, 2] = fill_color[2]
        #
        if fill_color is None:
            self.__img_mat[:, :, 3] = 0
        else:
            self.__img_mat[:, :, 3] = 255

    def get_img_mat(self):
        """get img mat"""
        return self.__img_mat.copy()

    def get_img_shape(self):
        """get shape"""
        return self.__img_mat.shape

    def get_alpha_layer(self):
        """get alpha layer"""
        return self.__img_mat[:, :, 3].copy()

    def get_assign_color_mask(self, assign_color):
        """get make by assign color"""
        mask = (self.__img_mat[:, :, 0] == assign_color[0]) & \
               (self.__img_mat[:, :, 1] == assign_color[1]) & \
               (self.__img_mat[:, :, 2] == assign_color[2])
        return mask

    def get_assign_value_mask(self, value, assign_index):
        """get where have assign value"""
        mask = (self.__img_mat[:, :, assign_index] == value)
        return mask

    def set_img_mat(self, image_mat):
        """set img mat value, value type should be np.uint8"""
        if not isinstance(image_mat, np.ndarray):
            raise TypeError('image_mat should be ndarry')

        if not image_mat.dtype == np.uint8:
            raise TypeError('image_mat type should be uint8')

        # todo checck shape

        self.__img_mat = image_mat

    def set_alpha_layer(self, alpha_layer):
        """set alpha layer"""
        if not isinstance(alpha_layer, np.ndarray):
            raise TypeError

        if not alpha_layer.dtype == np.uint8:
            raise TypeError

        if self.__img_mat[:, :, 3].shape != alpha_layer.shape:
            raise ValueError
        self.__img_mat[:, :, 3] = alpha_layer

    def set_assign_layer_value_by_mask(self, layer_index, mask, assign_value):
        """set alpha layer by function"""
        assign_layer = self.__img_mat[:, :, layer_index]
        assign_layer[mask] = assign_value
        self.__img_mat[:, :, layer_index] = assign_layer

    def extend_to_range(self, extend_range, fill_color=(255, 255, 255)):
        """
        扩展大小，正数是扩展，0是不变，不能有负数
        :param extend_range: 左，右，下，上
        :param fill_color: 填充色
        :return: image_mat
        """
        # todo rename
        row_num, col_num = self.__img_mat.shape[:2]
        image_mat_new = np.zeros(
            (row_num + extend_range[2] + extend_range[3], col_num + extend_range[0] + extend_range[1], 4),
            dtype=np.uint8)
        # 填充色
        if fill_color is None:
            image_mat_new[:, :, :] = 0
        else:
            image_mat_new[:, :, 0] = fill_color[0]
            image_mat_new[:, :, 1] = fill_color[1]
            image_mat_new[:, :, 2] = fill_color[2]
            image_mat_new[:, :, 3] = 255

        # 贴上原矩阵
        row_num, col_num = image_mat_new.shape[:2]
        image_mat_new[extend_range[3]:row_num - extend_range[2], extend_range[0]:col_num - extend_range[1],
        :] = self.__img_mat
        self.__img_mat = image_mat_new

    def extend_by_ratio(self, extend_ratio, fill_color=(255, 255, 255)):
        """根据比例对要素进行扩展,左增加row_num的倍数，右，下增加col_num的倍数"""
        # todo rename
        row_num, col_num = self.__img_mat.shape[:2]
        # 上下左右增加的数目
        left_extend = int(col_num * extend_ratio[0])
        right_extend = int(col_num * extend_ratio[1])
        bottom_extend = int(row_num * extend_ratio[2])
        top_extend = int(row_num * extend_ratio[3])
        # 新的画布
        extend_rows = row_num + bottom_extend + top_extend  #
        extend_cols = col_num + left_extend + right_extend  #
        image_mat_new = np.zeros((extend_rows, extend_cols, 4), dtype=np.uint8)
        # 填充色
        if fill_color is None:
            image_mat_new[:, :, :] = 0
        else:
            image_mat_new[:, :, 0] = fill_color[0]
            image_mat_new[:, :, 1] = fill_color[1]
            image_mat_new[:, :, 2] = fill_color[2]
            image_mat_new[:, :, 3] = 255

        # 贴上原矩阵
        row_num, col_num = image_mat_new.shape[:2]
        image_mat_new[top_extend:extend_rows - bottom_extend, left_extend:col_num - right_extend, :] = self.__img_mat
        self.__img_mat = image_mat_new

    def cut_border_in_assign_color(self, border_color=(255, 255, 255)):
        """
        去掉指定边框，可以指定边框颜色
        :param border_color:
        :return: image_mat
        """
        # 找到 image_mat 中值不为指定边框色的区域
        mask = np.logical_and(np.logical_and(self.__img_mat[:, :, 0] == border_color[0],
                                             self.__img_mat[:, :, 1] == border_color[1]),
                              self.__img_mat[:, :, 2] == border_color[2])
        # 找到区域的上下左右边界
        x, y = zip(*np.argwhere(mask == False))  # FIXME 这边需要进行修正
        self.__img_mat = self.__img_mat[min(x):max(x) + 1, min(y):max(y) + 1]

    def convert_to_assign_shape(self, assign_size):
        """图像缩放至指定的长宽，调用 Image 中的 resize 函数"""
        # 转为 image
        img = Image.fromarray(self.__img_mat)
        # 改变长宽
        img = img.resize(assign_size)
        # 转为矩阵
        self.__img_mat = np.array(img)

    def show_img(self):
        """将图像展示出来，在测试的时候很需要"""
        plt.imshow(self.__img_mat)
        plt.show()

    def save_to_image(self, save_path):
        """保存为图像"""
        if str(save_path).endswith('.png'):
            img = Image.fromarray(self.__img_mat)
        elif str(save_path).endswith('.jpg'):
            img = Image.fromarray(self.__img_mat[:, :, :3])
        else:
            raise TypeError('only support jpg and png')
        img.save(save_path)

    def cut_img_to_N_N(self, num_x, num_y, save_dir, augment_parameter=None):
        """将图像裁剪为 N*N 块"""

        # 设置扩展比率
        if augment_parameter is None:
            augment_parameter = [0, 0, 0, 0]

        height, width = self.__img_mat.shape[:2]
        each_height = int(height/float(num_x))
        each_width = int(width/float(num_y))
        img_name = os.path.split(self.__img_path)[1][:-4]

        for i in range(num_x):
            for j in range(num_y):
                each_range = [i*each_height, j*each_width, (i+1)*each_height, (j+1)*each_width]
                new_range = self.region_augment(each_range, (height, width), augment_parameter)
                xmin, ymin, xmax, ymax = new_range

                # 不丢弃图像化最后的几行像素
                if i == num_x-1: xmax = height
                if j == num_y-1: ymax = width

                new_img = self.__img_mat[xmin:xmax, ymin:ymax, :]
                img = Image.fromarray(new_img[:,:,:3])
                each_save_path = os.path.join(save_dir, "{0}_{1}_{2}.jpg".format(img_name, i, j))
                img.save(each_save_path)

    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __get_com_color(image_mat_a, image_mat_b):
        """在有透明度的情况下，得到重叠的颜色"""
        #  参考 https://www.baidu.com/link?url=FwnJAkG_TZpC1YnFNsTDj6_oVADGfZBG4IMliNLstm1Itix33MbbnGZnSKk-vP4WLJa5Ef-UmXsPF-coHY1d-c1F-K5zB7pEB3VB8WQtaFS&wd=&eqid=e8c48848000477d8000000055d04b866

        # 画布是 image_mat_b

        # FIXME 里面矩阵运算部分可以简化，暂时没想到什么好的方法，矩阵乘法那一块不是很熟

        alpha_a = (image_mat_a[:, :, 3] / 255.0).astype(np.float16)
        alpha_b = (image_mat_b[:, :, 3] / 255.0).astype(np.float16)

        rgb_a = image_mat_a[:, :, :3].astype(np.float) / 255
        rgb_b = image_mat_b[:, :, :3].astype(np.float) / 255

        com_alpha = (((alpha_a + alpha_b) - alpha_a * alpha_b) * 255).astype(np.uint8)  # 混合后的透明度

        com_rgb_a_r = rgb_a[:, :, 0] * alpha_a * ((1 - alpha_b) / (alpha_a + alpha_b - alpha_a * alpha_b))
        com_rgb_b_r = rgb_b[:, :, 0] * (alpha_b / (alpha_a + alpha_b - alpha_a * alpha_b))

        com_rgb_a_g = rgb_a[:, :, 1] * alpha_a * ((1 - alpha_b) / (alpha_a + alpha_b - alpha_a * alpha_b))
        com_rgb_b_g = rgb_b[:, :, 1] * (alpha_b / (alpha_a + alpha_b - alpha_a * alpha_b))

        com_rgb_a_b = rgb_a[:, :, 2] * alpha_a * ((1 - alpha_b) / (alpha_a + alpha_b - alpha_a * alpha_b))
        com_rgb_b_b = rgb_b[:, :, 2] * (alpha_b / (alpha_a + alpha_b - alpha_a * alpha_b))

        image_mat = np.zeros((image_mat_a.shape[0], image_mat_a.shape[1], 4), dtype=np.uint8)
        image_mat[:, :, 0] = ((com_rgb_a_r + com_rgb_b_r) * 255).astype(np.uint8)
        image_mat[:, :, 1] = ((com_rgb_a_g + com_rgb_b_g) * 255).astype(np.uint8)
        image_mat[:, :, 2] = ((com_rgb_a_b + com_rgb_b_b) * 255).astype(np.uint8)
        image_mat[:, :, 3] = com_alpha

        return image_mat

    @staticmethod
    def __create_canvas(row_num, col_num, fill_color=(255, 255, 255), alpha_value=None):
        """新建画布, y,x ==> img_mat"""
        rect_shape = np.ones((row_num, col_num, 4), dtype=np.uint8) * 255
        # 设置颜色
        rect_shape[:, :, 0] = fill_color[0]
        rect_shape[:, :, 1] = fill_color[1]
        rect_shape[:, :, 2] = fill_color[2]
        # 设置透明图层
        if alpha_value is None:
            rect_shape[:, :, 3] = 0
        else:
            rect_shape[:, :, 3] = alpha_value
        return rect_shape

    @staticmethod
    def get_img_mat_from_file(image_path):
        """
        图片转为图片矩阵
        :param image_path: 图片路径，str
        :return: image_mat
        """
        img = Image.open(image_path)
        image_mat = np.asarray(img, dtype=np.uint8)
        # 矩阵维度
        if image_mat.ndim == 2:
            alpha_mat = np.ones_like(image_mat) * 255
            # return np.array([image_mat.copy(), image_mat.copy(), image_mat.copy(), alpha_mat], dtype=np.uint8)
            image_mat = np.rollaxis(np.tile(image_mat, (4, 1, 1)), 0, 3)  # 单波段，转为多波段, TODO 整理下来
            image_mat[:, :, 3] = alpha_mat
            return image_mat
        else:
            # 不存在 alpha 图层, alpha 图层设置为 全是 255 的矩阵
            if image_mat.shape[2] == 3:
                alpha_mat = np.ones_like(image_mat[:, :, 1]) * 255
                image_mat = np.array(
                    [image_mat[:, :, 0].copy(), image_mat[:, :, 1].copy(), image_mat[:, :, 2].copy(), alpha_mat],
                    dtype=np.uint8)  # 变为 (4,x,y)
                image_mat = np.rollaxis(image_mat[[0, 1, 2, 3], :, :], 0, 3)  # 变为 (x,y,4)
                return image_mat
            elif image_mat.shape[2] == 4:
                return image_mat

    @staticmethod
    def get_rand_color(mode='RGB'):
        """得到随机的颜色"""
        if mode.upper() == 'RGB':
            return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        elif mode.upper() == 'GRAY':
            value_uint8 = random.randint(0, 255)
            return value_uint8, value_uint8, value_uint8

    @staticmethod
    def create_word_img(word, word_color=(0, 0, 0), word_size=100, font_type='simfang.ttf',
                        background_color=None, is_horizontal=True, image_font_truetype=None, is_cut_border=True):
        """得到文字，这个很麻烦，很多的机器上解决不了，尝试解决"""
        # ------------------------------
        # 如果水平排列
        if is_horizontal:
            image_shape = (word_size, len(word) * word_size)
        else:
            image_shape = (len(word) * word_size, word_size)
            word = '\n'.join(word)

        word_loc = (0, 0)  # 文字起始位置
        # ------------------------------
        row_num, col_num = image_shape
        # 设置所使用的字体
        if image_font_truetype:
            font = ImageFont.truetype(r"{0}\{1}".format(image_font_truetype, font_type), word_size)
        else:
            font = ImageFont.truetype(r"C:\Windows\Fonts\{0}".format(font_type), word_size)
        # 创建画布
        if background_color is None:
            img = Image.fromarray(ImageUtil.__create_canvas(row_num, col_num))
        else:
            img = Image.fromarray(ImageUtil.__create_canvas(row_num, col_num,
                                                            fill_color=background_color, alpha_value=255))
        # 画图
        draw = ImageDraw.Draw(img)
        draw.text(word_loc, word, word_color, font=font)  # 设置文字位置/内容/颜色/字体
        image_mat = np.array(img, dtype=np.uint8)  # 得到矩阵

        word_img = ImageUtil()
        word_img.set_img_mat(image_mat)

        # 去掉矩阵外面一圈边框
        if is_cut_border:
            if background_color is None:
                word_img.cut_border_in_assign_color(border_color=(255, 255, 255))
            else:
                word_img.cut_border_in_assign_color(border_color=background_color)
        return word_img

    @staticmethod
    def region_augment(region_rect, img_size, augment_parameter=None):
        """上下左右指定扩增长宽的比例, augment_parameter, 左右上下"""

        if augment_parameter is None:
            augment_parameter = [0.6, 0.6, 0.1, 0.1]

        widht, height = img_size
        x_min, y_min, x_max, y_max = region_rect
        region_width = int(x_max - x_min)
        region_height = int(y_max - y_min)
        new_x_min = x_min - int(region_width * augment_parameter[0])
        new_x_max = x_max + int(region_width * augment_parameter[1])
        new_y_min = y_min - int(region_height * augment_parameter[2])
        new_y_max = y_max + int(region_height * augment_parameter[3])

        new_x_min = max(0, new_x_min)
        new_y_min = max(0, new_y_min)
        new_x_max = min(widht, new_x_max)
        new_y_max = min(height, new_y_max)

        return (new_x_min, new_y_min, new_x_max, new_y_max)

    # ---------------------------------------- need repair -------------------------------------------------------------

    @staticmethod
    def create_shape_rect(row_num, col_num, fill_color=(255, 255, 255)):
        """新增矩形"""
        # 创建画布矩阵
        rect_shape = np.ones((row_num, col_num, 4), dtype=np.uint8) * 255
        # 设置颜色
        rect_shape[:, :, 0] = fill_color[0]
        rect_shape[:, :, 1] = fill_color[1]
        rect_shape[:, :, 2] = fill_color[2]
        # 设置透明图层
        if fill_color is None:
            rect_shape[:, :, 3] = 0
        else:
            rect_shape[:, :, 3] = 255

        rect_img = ImageUtil()
        rect_img.set_img_mat(rect_shape)
        return rect_img

    @staticmethod
    def create_shape_circle(radius, fill_color=(255, 255, 255)):
        """画圆"""
        # FIXME 这个因为只有一个颜色，所以画出的圆很粗糙，重新使用 Image 自带的画图画圆

        # FIXME 根据距离的大小设置颜色（颜色的纯度）

        # 创建画布矩阵
        circle_shape = np.ones((radius * 2, radius * 2, 4), dtype=np.uint8) * 255
        # 得到计算需要的行列矩阵，排序矩阵的计算并变形？
        range_mat = np.array(range((radius * 2) ** 2)).reshape((radius * 2, radius * 2))
        row_mat = range_mat / (radius * 2)
        col_mat = range_mat % (radius * 2)
        # 计算得到圆掩膜（mask）
        mask = (row_mat - radius) ** 2 + (col_mat - radius) ** 2 < radius ** 2
        # 根据掩膜得到需要的shape, 加上底色
        circle_shape[mask, 0] = fill_color[0]
        circle_shape[mask, 1] = fill_color[1]
        circle_shape[mask, 2] = fill_color[2]
        circle_shape[~mask, 3] = 0  # 将圆以外的区域设置为透明

        circle_img = ImageUtil()
        circle_img.set_img_mat(circle_shape)
        return circle_img

    @staticmethod
    def create_shape_ellipse(row_r, col_r, fill_color=(255, 255, 255), background_color=(255, 255, 255)):
        """画椭圆"""
        # 从矩阵创建
        if background_color is None:
            ellipse = Image.fromarray(ImageUtil.__create_canvas(row_r, col_r))
        else:
            ellipse = Image.fromarray(ImageUtil.create_shape_rect(row_r, col_r, fill_color=background_color))
        draw = ImageDraw.Draw(ellipse)
        # 画椭圆
        draw.ellipse((0, 0, row_r, col_r), fill=fill_color)
        #
        ellipse_img = ImageUtil()
        ellipse_img.set_img_mat(np.array(ellipse, dtype=np.uint8))
        return ellipse_img

    @staticmethod
    def create_shape_polygon(point_list, fill_color=(100, 100, 100), background_color=(255, 255, 255)):
        """画多边形"""
        # 从矩阵创建
        x, y = zip(*point_list)
        if background_color is None:
            polygon = Image.fromarray(ImageUtil.__create_canvas(max(y), max(x)))
        else:
            polygon = Image.fromarray(ImageUtil.create_shape_rect(max(y), max(x), fill_color=background_color))
        # 画图
        draw = ImageDraw.Draw(polygon)
        draw.polygon(point_list, fill=fill_color)

        ploygon_img = ImageUtil()
        ploygon_img.set_img_mat(np.array(polygon, dtype=np.uint8))
        return ploygon_img



if __name__ == "__main__":

    a = ImageUtil(r"C:\Users\14271\Desktop\del\test.jpg")
    a.cut_img_to_N_N(15, 5, r"C:\Users\14271\Desktop\cut_res")
