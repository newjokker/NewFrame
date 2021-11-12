# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import shutil
import os
import datetime


class FileOperationUtil(object):
    """文件操作类"""

    @staticmethod
    def delete_folder(dir_path):
        """删除一个路径下的所有文件"""
        # todo 这个一般都是最后一步，所以如何解决删除文件报错的问题，可能是文件被程序在占用
        shutil.rmtree(dir_path)

    @staticmethod
    def create_folder(folder_path):
        """
        如果文件夹不存在创建文件夹，如果文件夹存在选择是否清空文件夹
        :param folder_path:
        :return: 0: 已存在，未创建  1：不存在，已创建
        """
        if os.path.isdir(folder_path):
            return 1
        else:
            os.makedirs(folder_path)
            return 0

    @staticmethod
    def bang_path(file_path):
        """
        将文件名给bang开，这样省的麻烦，只能是文件地址，文件夹不好 bang 开
        :param file_path: 文件路径
        :return: folder_path, file_name, file_extenction
        """
        if not os.path.isfile(file_path):
            raise EOFError ("需要输入文件路径，而不是文件夹路径或者其他")

        #  (1) 得到文件夹路径
        folder_path = os.path.split(file_path)
        # （2）得到文件名
        file_name = os.path.splitext(folder_path[1])[0]
        # （3）得到后缀
        file_suffix = os.path.splitext(folder_path[1])[1]

        return folder_path[0], file_name, file_suffix

    @staticmethod
    def re_all_file(file_path, func=None):
        """
         返回文件夹路径下的所有文件路径（搜索文件夹中的文件夹）
         传入方法对文件路径进行过滤
        :param file_path:
        :param func: 用于筛选路径的方法
        :return:
        """

        # 【1】判断输入参数
        if not os.path.isdir(file_path):
            print(" 不是文件夹路径 ")
            raise EOFError

        result = []
        for i, j, k in os.walk(file_path):
            for each in k:
                abs_path = i + os.sep + each
                if func is None:  # is 判断是不是指向同一个东西
                    result.append(abs_path)
                else:
                    # 使用自定义方法对文件进行过滤
                    if func(abs_path):
                        result.append(os.path.join(i, each))
        return result

    @staticmethod
    def get_file_describe_dict(file_path):
        """文件描述，返回需要的文件描述信息"""
        desrb = {'_size_': str(round(float(os.path.getsize(file_path)) / 1024 ** 2, 4)) + ' M',
                 'a_time': datetime.datetime.utcfromtimestamp(os.path.getatime(file_path)),
                 'c_time': datetime.datetime.utcfromtimestamp(os.path.getctime(file_path)),
                 'm_time': datetime.datetime.utcfromtimestamp(os.path.getmtime(file_path))}
        return desrb

    @staticmethod
    def move_file_to_folder(file_path_list, assign_folder, is_clicp=False):
        """将列表中的文件路径全部拷贝或者剪切到指定文件夹下面，is_clip 是否剪切，否就是复制"""
        for each_file_path in file_path_list:
            # 过滤掉错误的文件路径
            if not os.path.isfile(each_file_path):
                print("file not exist : {0}".format(each_file_path))
                continue
            #
            new_file_path = os.path.join(assign_folder, os.path.split(each_file_path)[1])
            if is_clicp:
                shutil.move(each_file_path, new_file_path)
            else:
                shutil.copyfile(each_file_path, new_file_path)

    # ------------------------------------ need repair -----------------------------------------------------------------

    @staticmethod
    def get_father_path(str_temp):
        """ 查找父文件夹，mac 和 windows 环境下都能运行
        input:
            str_temp: str
        output:
            str_temp 的父级文件夹，str
        """
        # fixme 有对应的函数的
        # 去掉末尾的 '\' 和 '/'
        str_temp = str_temp.rstrip(r'/')
        str_temp = str_temp.rstrip(r'\\')
        return os.path.split(str_temp)[0]

    @staticmethod
    def clear_empty_folder():
        """删除空文件夹"""
        pass



if __name__ == '__main__':

    file_type_dict = {"_fzc_": [],
                      "_qx_": [],
                      "_other_": [],
                      "_zd_": []}

    file_folder = r'C:\data\fzc_优化相关资料\dataset_fzc\015_防振锤准备使用faster训练_在原图上标注\002_截为小图，查问题'

    for i in FileOperationUtil.re_all_file(file_folder):
        for each_name in file_type_dict:
            if each_name in i:
                file_type_dict[each_name].append(i)
                continue

    for each_name in file_type_dict:
        folder_path = os.path.join(file_folder, each_name.strip('_'))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        #
        FileOperationUtil.move_file_to_folder(file_type_dict[each_name], folder_path, is_clicp=True)


    print("OK")




