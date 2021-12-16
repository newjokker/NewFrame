# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import numpy as np
import cv2
from skimage import io, data
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import exifread
import re
import json
import requests


# opencv 安装 conda install -c menpo opencv3  即可
#整理常用的库函数, 参考 ：https://blog.csdn.net/weixin_42029090/article/details/80618208


class ImageProcessingUtil(object):

    @staticmethod
    def sp_noise(image, prob):
        """添加椒盐噪声, prob:噪声比例"""
        image = image.copy()  # 取原矩阵的深拷贝

        prob = prob / 2.0  # 信噪比 ∈ （0,1）所以先除以 2
        thres = 1 - prob
        if thres < prob:
            thres, prob = prob, thres

        m, n = image.shape
        mask = np.random.rand(m, n)  # 返回与 image 通行列的随机数矩阵
        image[mask < prob] = 0
        image[mask > thres] = 255
        return image

    @staticmethod
    def gasuss_noise(image, mean=0, var=0.1):
        """ 添加高斯噪声, mean : 均值, var : 方差"""
        image = np.array(image / 255.0, dtype=float)  # 设置值域为 （0, 1）
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        out = np.clip(out, 0, 1.0)
        out = np.uint8(out * 255)
        return out

    # --------------------------------- open cv -------------------------------------

    @staticmethod
    def img_read(img_path):
        """读取图像"""
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)    # 读取路径中含有中文的图像，返回矩阵

    @staticmethod
    def img_read_as_gray(img_path):
        """将图像读取为灰色"""
        return cv2.imread(r'C:\Users\74722\Desktop\test.jpg', cv2.IMREAD_GRAYSCALE)  # cv2 以灰色模式读取图像

    @staticmethod
    def img_split(img_path):
        """图像的分波段"""
        bgr = ImageProcessingUtil.img_read(img_path)
        b, g, r = cv2.split(bgr)
        return b, g, r

    @staticmethod
    def img_merge(r, g, b, save_path):
        """将三通道的矩阵压缩成图片"""
        img = cv2.merge((b, g, r))
        cv2.imwrite(save_path, img)

    @staticmethod
    def find_min_area_Rect_from_point(points):
        """找到点集的最小外接矩形"""
        # 参考 https://www.jianshu.com/p/6bde79df3f9d
        min_rect = cv2.minAreaRect(np.array(points))    # 找到最小外接矩形的信息，（矩形的中心点，矩阵的长宽，矩形旋转的角度）
        box = cv2.boxPoints(min_rect)                   # 将最小外接矩形的信息进行规范化，返回四个点坐标
        return box

    @staticmethod
    def img_convolution(img_data):
        """图像二维卷积"""
        core = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.float32) / 9   # 卷积核
        res = cv2.filter2D(img_data, -1, core)                                      # 卷积
        return res

    @staticmethod
    def img_convolution_in_assign_method(img_data, assign_method=None):
        """使用自定义函数对图像进行卷积"""
        # import scipy.ndimage as ndi
        if assign_method is None:
            assign_method = lambda x: np.percentile(x, 50)      # 找到中位数
        res = ndi.generic_filter(img_data, assign_method, size=3)
        return res

    @staticmethod
    def bytescale(data, cmin=None, cmax=None, high=255, low=0):
        """图像进行归一化，图像被归一化到 [cmin, cmax]"""
        # 复写函数 ： from scipy.misc import bytescale
        if data.dtype == np.uint8:  # 当输入数据的类型是 uint8 的时候，就直接返回不做操作
            return data

        if high > 255:
            raise ValueError("`high` should be less than or equal to 255.")
        if low < 0:
            raise ValueError("`low` should be greater than or equal to 0.")
        if high < low:
            raise ValueError("`high` should be greater than or equal to `low`.")

        if cmin is None:
            cmin = data.min()
        if cmax is None:
            cmax = data.max()

        cscale = cmax - cmin
        if cscale < 0:
            raise ValueError("`cmax` should be larger than `cmin`.")
        elif cscale == 0:
            cscale = 1

        scale = float(high - low) / cscale
        bytedata = (data - cmin) * scale + low
        return (bytedata.clip(low, high) + 0.5).astype(np.uint8)  # 给定一个区间，该区间外的值被剪切到该区间，最后为什么要加 0.5 ？

    @staticmethod
    def harris(img_data):
        """特征检测"""
        # 参考 ： opencv3 计算机视觉 P84
        gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 23, 0.04)
        # img_data[dst > 0.01 * dst.max()] = [0, 0, 255]  # 超过阈值转为红色，用于查看结果
        return dst

    @staticmethod
    def tuxiangzengqiang():
        """图像增强"""

        # todo 进行整理

        from skimage import data
        from skimage import exposure, img_as_float
        import matplotlib.pyplot as plt

        # ------------------- 灰度变换 -----------------------------------------------------------------------------------------
        imge5 = img_as_float(data.coffee())  # 把图像的像素值转换为浮点数
        gam1 = exposure.adjust_gamma(imge5, 2)  # 使用伽马调整，第二个参数控制亮度，大于1增强亮度，小于1降低。
        log1 = exposure.adjust_log(imge5, 0.7)  # 对数调整

        # 用一行两列来展示图像
        plt.subplot(1, 3, 1)
        # plt.imshow(imge5, plt.cm.gray)
        plt.imshow(imge5)

        plt.subplot(1, 3, 2)
        # plt.imshow(gam1, plt.cm.gray)
        plt.imshow(gam1)

        plt.subplot(1, 3, 3)
        # plt.imshow(log1, plt.cm.gray)
        plt.imshow(log1)

        plt.show()

        return

    @staticmethod
    def zhifangtujunhenghua():
        """直方图均衡化"""
        from skimage import data, exposure
        import matplotlib.pyplot as plt

        # 直方图均衡化

        img6 = data.coffee()
        # 指定绘制的大小
        plt.figure("hist", figsize=(8, 8))

        # 把图像的二维数组按行转为一维数组，这样才能绘制直方图
        arr = img6.flatten()

        plt.subplot(2, 2, 1)
        plt.imshow(img6, plt.cm.gray)
        plt.subplot(2, 2, 2)
        # 绘制直方图
        plt.hist(arr, bins=256, normed=1, edgecolor='None', facecolor='red')

        # 对直方图进行均衡化
        img_c = exposure.equalize_hist(img6)
        arr_c = img_c.flatten()
        plt.subplot(2, 2, 3)
        plt.imshow(img_c, plt.cm.gray)
        plt.subplot(2, 2, 4)
        plt.hist(arr_c, bins=256, normed=1, edgecolor='None', facecolor='red')

        plt.show()

    @staticmethod
    def zishiyingzhifangtujuhenghua():
        """自适应直方图均衡化"""
        import cv2
        import numpy as np
        from PIL import Image
        import scipy
        import matplotlib.image as img

        # 参考 ： https://www.cnblogs.com/my-love-is-python/p/10405811.html

        data = img.imread(r'C:\Users\Administrator\Desktop\aaa.jpg')

        # cv2.equalizeHist(img)  # 表示进行直方图均衡化

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(
        10, 10))  # 用于生成自适应均衡化图像, # LAB 它是用数字化的方法来描述人的视觉感应 https://blog.csdn.net/denghecsdn/article/details/78031825
        claheB = clahe.apply(np.array(data, dtype=np.uint8))
        cv2.imwrite(r'C:\Users\Administrator\Desktop\123.png', data)

        # FIXME 可以先调整为 LAB 颜色空间，然后对 L（light）进行直方图自适应均衡化，就能让整个图片的亮度保持均衡

        # 参数说明：clipLimit : 颜色对比度的阈值， titleGridSize : 进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作


class GetPictureInfo():
    """拿到照片的信息"""

    @staticmethod
    def latitude_and_longitude_convert_to_decimal_system(*arg):
        """
        经纬度转为小数, param arg:
        :return: 十进制小数
        """
        return float(arg[0]) + (
                    (float(arg[1]) + (float(arg[2].split('/')[0]) / float(arg[2].split('/')[-1]) / 60)) / 60)

    @staticmethod
    def find_GPS_image(pic_path):
        GPS = {}
        date = ''
        with open(pic_path, 'rb') as f:
            tags = exifread.process_file(f)
            for tag, value in tags.items():
                if re.match('GPS GPSLatitudeRef', tag):
                    GPS['GPSLatitudeRef'] = str(value)
                elif re.match('GPS GPSLongitudeRef', tag):
                    GPS['GPSLongitudeRef'] = str(value)
                elif re.match('GPS GPSAltitudeRef', tag):
                    GPS['GPSAltitudeRef'] = str(value)
                elif re.match('GPS GPSLatitude', tag):
                    try:
                        match_result = re.match('\[(\w*),(\w*),(\w.*)/(\w.*)\]', str(value)).groups()
                        GPS['GPSLatitude'] = int(match_result[0]), int(match_result[1]), int(match_result[2])
                    except:
                        deg, min, sec = [x.replace(' ', '') for x in str(value)[1:-1].split(',')]
                        GPS['GPSLatitude'] = GetPictureInfo.latitude_and_longitude_convert_to_decimal_system(deg, min,
                                                                                                             sec)
                elif re.match('GPS GPSLongitude', tag):
                    try:
                        match_result = re.match('\[(\w*),(\w*),(\w.*)/(\w.*)\]', str(value)).groups()
                        GPS['GPSLongitude'] = int(match_result[0]), int(match_result[1]), int(match_result[2])
                    except:
                        deg, min, sec = [x.replace(' ', '') for x in str(value)[1:-1].split(',')]
                        GPS['GPSLongitude'] = GetPictureInfo.latitude_and_longitude_convert_to_decimal_system(deg, min,
                                                                                                              sec)
                elif re.match('GPS GPSAltitude', tag):
                    GPS['GPSAltitude'] = str(value)
                elif re.match('.*Date.*', tag):
                    date = str(value)
        return {'GPS_information': GPS, 'date_information': date}

    @staticmethod
    def find_address_from_GPS(GPS):
        """
        使用Geocoding API把经纬度坐标转换为结构化地址。
        :param GPS:
        :return:
        """
        secret_key = 'zbLsuDDL4CS2U0M4KezOZZbGUY9iWtVf'
        if not GPS['GPS_information']:
            return '该照片无GPS信息'
        lat, lng = GPS['GPS_information']['GPSLatitude'], GPS['GPS_information']['GPSLongitude']
        baidu_map_api = "http://api.map.baidu.com/geocoder/v2/?ak={0}&callback=renderReverse&location={1},{2}s&output=json&pois=0".format(
            secret_key, lat, lng)
        response = requests.get(baidu_map_api)
        content = response.text.replace("renderReverse&&renderReverse(", "")[:-1]
        baidu_map_address = json.loads(content)
        formatted_address = baidu_map_address["result"]["formatted_address"]
        province = baidu_map_address["result"]["addressComponent"]["province"]
        city = baidu_map_address["result"]["addressComponent"]["city"]
        district = baidu_map_address["result"]["addressComponent"]["district"]
        return formatted_address, province, city, district


if __name__ == '__main__':
    coins = data.coins()
    coins_sp = ImageProcessingUtil.sp_noise(coins, 0.1)
    coins_gasuss = ImageProcessingUtil.gasuss_noise(coins)

    plt.subplot(311), plt.imshow(coins, cmap='gray')
    plt.subplot(312), plt.imshow(coins_sp, cmap='gray')
    plt.subplot(313), plt.imshow(coins_gasuss, cmap='gray')
    plt.show()
