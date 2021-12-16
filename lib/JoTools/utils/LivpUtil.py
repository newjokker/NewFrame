# -*- coding: utf-8  -*-
# -*- author: jokker -*-


# 处理 iphone 拍照后得到的 livp 格式的图片，解压为 heic + mov 并进一步解析为 jpg

import os
from ..utils.ZipUtil import ZipUtil
import whatimage
import pyheif
import traceback
from PIL import Image


class LivpUtil():

    @staticmethod
    def unzip_to_heic(file_path, save_folder):
        """将 livp 文件加压为 heic + mov 保存到指定的文件夹，返回保存的 heic 文件的路径"""
        if not str(file_path).endswith(".livp") or not os.path.isfile(file_path):
            raise ValueError("error livp file path")
        # 拿到解析后的所有文件路径
        file_path_list = ZipUtil.unzip_file(file_path, save_folder)
        # 返回 。heic 文件的路径
        return file_path_list[0] if file_path_list[0].endswith('.heic') else file_path_list[1]

    @staticmethod
    def decodeImage(bytesIo, save_path):
        try:
            fmt = whatimage.identify_image(bytesIo)
            # print('fmt = ', fmt)
            if fmt in ['heic']:
                i = pyheif.read_heif(bytesIo)
                # print('i = ', i)
                # print('i.metadata = ', i.metadata)
                pi = Image.frombytes(mode=i.mode, size=i.size, data=i.data)
                # print('pi = ', pi)
                pi.save(save_path, format="jpeg")
        except:
            traceback.print_exc()

    @staticmethod
    def read_image_file_rb(file_path):
        with open(file_path, 'rb') as f:
            file_data = f.read()
        return file_data

    @staticmethod
    def heic_to_jpg(heic_file_path, save_path):
        """heic 文件转为 jpg 文件"""
        data = LivpUtil.read_image_file_rb(heic_file_path)
        LivpUtil.decodeImage(data, save_path)


if __name__ == "__main__":

    livp_dir = r"/home/ldq/del/heic"
    temp_folder = r"/home/ldq/del/livp"
    save_folder = r"/home/ldq/del/jpg_res"


    for each_name in os.listdir(livp_dir):
        file_path = os.path.join(livp_dir, each_name)
        # 解压为 .heic 文件
        uzip_path = LivpUtil.unzip_to_heic(file_path, temp_folder)
        # 继续解析为 jpg 文件
        uzip_name = os.path.split(uzip_path)[1]
        save_path = os.path.join(save_folder, uzip_name[:-4] + 'jpg')
        LivpUtil.heic_to_jpg(uzip_path, save_path)