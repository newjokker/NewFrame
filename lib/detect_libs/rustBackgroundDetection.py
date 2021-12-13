
import os,sys
import cv2

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')

import json
import numpy as np
import warnings
import random
from ..detect_utils.tryexcept import try_except
from .backgroundDetection import BackgroundDetection


class RustBackgroundDetection(BackgroundDetection):
    def __init__(self, args, objName, scriptName):
        super(RustBackgroundDetection, self).__init__(args, objName, scriptName)
        self.resizedImgPath = self.getTmpPath('resizedImg')

    @try_except()
    def cal_color_single(self,img):
        color_angle = []
        img = img / img.max()  # 首先对图像进行归一化
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pix_BGR = img[i, j, :]
                if not np.any(pix_BGR):
                    pass
                else:
                    # 颜色角度：方向角d，高度角h
                    if pix_BGR[0] == 0:
                        d = np.pi / 2
                    else:
                        tan_d = pix_BGR[1] / pix_BGR[0]
                        d = np.arctan(tan_d)
                    if pix_BGR[0] == 0 and pix_BGR[1] == 0:
                        h = np.pi / 2
                    else:
                        tan_h = pix_BGR[2] / np.sqrt(pix_BGR[0] ** 2 + pix_BGR[1] ** 2)
                        h = np.arctan(tan_h)

                    # 为防止奇异值，这里使用了一个诡异的特性:nan ！= nan。应该不会触发。若不慎触发后会给出警告。
                    if h == h and d == d:
                        color_angle.append([d, h])
                    else:
                        warnings.warn('角度出现奇异值')
        return color_angle

    @try_except()
    def cal_rust(self,rate, safe_X, safe_Y, safe_rate, angle):
        isRust_list = []
        for i in range(len(angle)):
            thres = random.random()
            if thres < rate:  # 像素点过多，看不出规律。需进行采样。
                h_hudu = angle[i, 0]  ##表示红色，X轴
                d_hudu = angle[i, 1]  ## Y轴
                h_jiaodu = h_hudu / np.pi * 180
                d_jiaodu = d_hudu / np.pi * 180
                ## 安全区域（非锈蚀）内
                isRust_value = 1 if (h_jiaodu < safe_X) and (d_jiaodu < safe_Y) else 0
                isRust_list.append(isRust_value)
        if len(isRust_list) > 0:
            noRust_rate = sum(isRust_list) / len(isRust_list)
            predict_name = 'clean' if noRust_rate > safe_rate else 'rust'
            # print('-->noRust_rate: {} ,predict_name: {} :'.format(noRust_rate, predict_name))
        else:
            predict_name = 'clean'
        return predict_name

    @try_except()
    def get_label(self):
        # 通过调用json标签，形成锈蚀区域的标注矩阵。
        black_base = np.zeros([901, 901])  # 黑背景
        label_json = os.path.join(lib_path,'rust_cls_libs', 'label.json')
        with open(label_json, 'r') as load_f:
            load_dict = json.load(load_f)

            # 读取完成之后，文件就可以关闭了，当缩进结束，自动调用 .close()方法，避免忘记关闭。
        dict_shape = load_dict['shapes']
        for area in dict_shape:
            points_list = area['points']
            points = np.array(points_list, dtype=np.int32)
            points[:, 1] = 900 - points[:, 1]  # 注意了，这一行价值5个小时。对json标注的纵坐标进行上下翻转，以符合图片的读入格式

            label = 5
            if area['label'] == 'normal':
                label = 1
                cv2.fillPoly(black_base, [points], label)
            elif area['label'] == 'rust_red':
                label = 2
                cv2.fillPoly(black_base, [points], label)
            elif area['label'] == 'rust_yellow':
                label = 3
                cv2.fillPoly(black_base, [points], label)
            elif area['label'] == 'others':
                label = 4
                cv2.fillPoly(black_base, [points], label)
            assert label != 5  # 防止奇怪的错误

        return black_base.astype(int)  # 得到了不同区域的标签矩阵。标签分别为0：背景，1：安全区，2：红锈区，3：黄锈区，4：其他区

    @try_except()
    def cal_rust_matrix(self,img_seg, area_label, rust_rate):  # 返回bool型

            # 这里必须要转换为float或其他可进行浮点运算的类型。图片默认为uint8类型，在乘除和开方时，进行的不是浮点运算。
            B = img_seg[:, :, 0].astype(float)  # 防止奇异值，在B通道上全部加1，应该不影响最终的结果
            B += 1
            G = img_seg[:, :, 1].astype(float)
            R = img_seg[:, :, 2].astype(float)

            # 方向角正切矩阵和高度角正切矩阵的计算
            d = np.arctan(G / B) / np.pi * 1800
            h = np.arctan(R / np.sqrt(G ** 2 + B ** 2)) / np.pi * 1800

            hs = h.reshape(h.size).astype(int)
            ds = d.reshape(d.size).astype(int)  # 展平，并转换为整型，后续将作为索引使用。
            img_color_space = np.zeros([901, 901], dtype=int)  # 表述该图像的色彩空间，高度角放在第一个维度

            # 剔除数组中的无关元素，将需要遍历的对象的长度大大缩短。
            idx0_h = np.where((hs < 200) | (hs > 700))
            hs_half_zero = np.delete(hs, idx0_h)
            ds_half_zero = np.delete(ds, idx0_h)  # 删除h里面label区域之外的
            idx0_d = np.where((ds_half_zero < 100) | (ds_half_zero > 700))  # 这里有问题。不知道是哪里的问题。禁止修改这句话。
            hs_none_zero = np.delete(hs_half_zero, idx0_d)
            ds_none_zero = np.delete(ds_half_zero, idx0_d)  # 删除d里面label区域之外的
            assert hs_none_zero.size == ds_none_zero.size, '非零数组长度不同'

            # 经过一波删减之后，可迭代对象长度下降到了1-2千。提速50%
            for i in range(hs_none_zero.size):
                img_color_space[ds_none_zero[i]][hs_none_zero[i]] += 1

            # 计算各个分区都被覆盖了多少空间。
            img_color_mask = np.array(img_color_space != 0).astype(int)
            color_in_area = area_label * img_color_mask

            # 各个区域的像素计数
            num_normal = img_color_space[np.where(color_in_area == 1)].sum()  # 安全
            num_red = img_color_space[np.where(color_in_area == 2)].sum()  # 红锈
            num_yellow = img_color_space[np.where(color_in_area == 3)].sum()  # 黄锈

            num_rust = num_red + num_yellow  # 通过这个赋值，可以神奇地提升一点速度。最大限度的缩减运算
            if num_rust == 0:  # 有的图，全部都被分割为背景，导致所有区域的数量都是0，这种图默认是干净的。
                clean = 'clean'
            else:
                if (num_rust / (num_rust + num_normal)) > rust_rate:
                    clean = 'rust'
                else:
                    clean = 'clean'

            return clean