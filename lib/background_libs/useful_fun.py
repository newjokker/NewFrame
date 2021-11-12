import numpy as np
import warnings
import cv2
import os
from tqdm import tqdm
# 有的图片全都是背景，出现了除数为0的情况。这里设置后，避免报出：RuntimeWarning: invalid value encountered in true_divide的警告
np.seterr(divide='ignore', invalid='ignore')


def cal_color(seg_dic) -> list:
    color_angle = []
    for img_name in tqdm(os.listdir(seg_dic)):
        img_path = seg_dic + '/' + img_name
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
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
                        warnings.warn('角度出现奇异值。图片名：{}'.format(img_name))
    return color_angle


# 该函数对检测速度的影响很小，几乎不占用计算时间。程序可以直接运行，此函数并未输出任何结果
def cal_dis(figure, Ljc_pix):
    # 两个超参数
    rust_color = [0, 0, 0]  # 在此定义锈蚀的标准色，需要进行归一化操作
    distance = 1  # 在此定于距离，需要进行实验来确定此距离，直接决定了锈蚀程度

    figure = figure / figure.max()  # 首先对图像进行归一化

    pix_all = figure.size  # 3个通道的全部的元素
    pix_1_channel = pix_all / 3  # 图片的面积
    # 三通道色值，统一运算，加速推理。最后再逐个遍历前景
    B_channel = figure[:, :, 0]
    G_channel = figure[:, :, 1]
    R_channel = figure[:, :, 2]

    # 思路：与标准锈蚀的颜色进行比对，通过在色彩空间中的距离来确定像素是否满足锈蚀条件
    rust_matrix = np.ones_like(figure)  # 生成锈蚀色矩阵
    rust_matrix[:, :, 0] = rust_matrix[:, :, 0] * rust_color[0]
    rust_matrix[:, :, 1] = rust_matrix[:, :, 1] * rust_color[1]
    rust_matrix[:, :, 2] = rust_matrix[:, :, 2] * rust_color[2]

    # 色彩空间的距离矩阵
    color_dis = np.sqrt((B_channel - rust_matrix[:, :, 0]) ** 2 + (G_channel - rust_matrix[:, :, 1]) ** 2 + (
            R_channel - rust_matrix[:, :, 2]) ** 2)

    # 然后数一下 color_dis矩阵中，有多少元素大于 distance。通过面积比即可得出锈蚀程度
    for i in range(Ljc_pix[0].size):  # 遍历所有前景像素
        idxx = Ljc_pix[0][i]
        idxy = Ljc_pix[1][i]
        pix = figure[idxx, idxy]  # 像素的BGR值

        pix_dis = color_dis[(idxx, idxy)]  # 该像素的距离
        pass


