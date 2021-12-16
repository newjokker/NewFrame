# -*- coding: utf-8  -*-
# -*- author: jokker -*-

"""用于打包和解压数据
参考： https://www.cnblogs.com/dancesir/p/9668967.html

* 虽然叫zipfile，但是除了zip之外，rar，war，jar这些压缩（或者打包）文件格式也都可以处理

"""

import zipfile
import os
from ..utils.FileOperationUtil import FileOperationUtil


class ZipUtil(object):

    @staticmethod
    def unzip_file(zip_file_path, save_folder):
        """将压缩文件解压 ，存放到对应的文件夹中去"""
        file_path_list = []
        f = zipfile.ZipFile(zip_file_path, 'r')
        for file_temp in f.namelist():
            f.extract(file_temp, save_folder)
            file_path_list.append(os.path.join(save_folder, file_temp))
        return file_path_list

    @staticmethod
    def zip_files(file_list, save_path, folder_name=None):
        """压缩文件"""
        # 提取每个文件的文件名
        if folder_name:
            arcname_list = list(map(lambda x: os.path.join(folder_name, os.path.split(x)[1]), file_list))  # 获取文件名
        else:
            arcname_list = list(map(lambda x: os.path.split(x)[1], file_list))  # 获取文件名

        with zipfile.ZipFile(save_path, "w", zipfile.zlib.DEFLATED) as zf:
            for file_index, file_path in enumerate(file_list):
                zf.write(file_path, arcname_list[file_index])

    @staticmethod
    def zip_folder(folder_path, save_zip_path, folder_name=None):
        """压缩文件夹"""
        if folder_name is None:  # 未指定名字的话，可以使用当前文件夹的名字
            folder_name = os.path.split(folder_path)[1]

        # 读取所有的文件路径
        file_list = []
        arcname_list = []
        folder_str_length = len(folder_path)
        for i, j, k in os.walk(folder_path):
            for each_file in k:
                abs_path = os.path.join(i, each_file)
                file_list.append(abs_path)
                arcname_list.append(folder_name + abs_path[folder_str_length:])  # 指定文件在压缩文件中的相对位置

        with zipfile.ZipFile(save_zip_path, "w", zipfile.zlib.DEFLATED) as zf:
            for file_index, each_file in enumerate(file_list):
                zf.write(each_file, arcname_list[file_index])


if __name__ == "__main__":


    folder_path = r"C:\Users\14271\Desktop\livp"

    for each_file in FileOperationUtil.re_all_file(folder_path, lambda x:str(x).endswith('.livp')):
        ZipUtil.unzip_file(each_file, r"C:\Users\14271\Desktop\livp\res")