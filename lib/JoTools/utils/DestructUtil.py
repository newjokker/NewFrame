# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import shutil
from .JsonUtil import JsonUtil

# todo 可以将文件存放在系统 tmp 目录中，这样就不会再包中留下痕迹

abs_path = os.path.abspath(__file__)
abs_dir = os.path.dirname(abs_path)
file_path = os.path.join(abs_dir, '.destruct.json')

del_count = 'del_count'
now_count = 'now_count'


class DestructUtil(object):

    @staticmethod
    def destruct_file_after_times(assign_file_path, assign_times):
        """在运行执行指定次数之后删除制定文件或者文件夹"""
        # 读取文件信息
        if os.path.exists(file_path):
            file_dict = JsonUtil.load_data_from_json_file(file_path)
        else:
            file_dict = {}
        # 缓存文件
        if not os.path.exists(assign_file_path):
            return
        # 可以执行次数的字典
        if assign_file_path not in file_dict:
            file_dict[assign_file_path] = {del_count: assign_times, now_count: 1}
        else:
            file_dict[assign_file_path][now_count] += 1
        # 判断文件是否要进行删除
        if file_dict[assign_file_path][now_count] >= file_dict[assign_file_path][del_count]:
            del file_dict[assign_file_path]
            JsonUtil.save_data_to_json_file(file_dict, file_path)
            if os.path.isfile(assign_file_path):
                os.remove(assign_file_path)
                return 0
            else:
                shutil.rmtree(assign_file_path)
                return 0
        JsonUtil.save_data_to_json_file(file_dict, file_path)
        # 返回还可以执行的次数
        return file_dict[assign_file_path][del_count] - file_dict[assign_file_path][now_count]

    @staticmethod
    def destruct_file_by_mix(assign_file_path):
        """通过将文件进行打乱的方式销毁文件，操作是可逆的，可以根据编码进行恢复"""
        pass

    @staticmethod
    def destruct_file_by_hide(assign_file_path):
        """通过将文件改变文件名后隐藏，对文件进行 删除"""
        pass

    @staticmethod
    def destruct_file_by_random(assign_file_path):
        """对文件随机进行删除"""
        pass


    














