# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import re
import numpy as np

"""
正则表达式的使用：
* re.compile
* re.findall
* match
"""


class StrUtil(object):

    def __init__(self):
        pass

    @staticmethod
    def split(string_to_split, split_character_list: list):
        """split string by, split_character_list ==> [] ==> ['**', '*', 'jok', '-']"""
        return re.split("|".join(split_character_list), string_to_split)

    @staticmethod
    def remove_assign_character(str_to_clean, characters_to_remove):
        """remove assign character"""
        for each_character in characters_to_remove:
            str_to_clean = str_to_clean.replace(each_character, '')
        return str_to_clean

    @staticmethod
    def match(match_str, compile_str):
        """if match"""
        # res = True if re.compile(compile_str).match(match_str) is not None else False
        return re.compile(compile_str).match(match_str)

    @staticmethod
    def find_all(find_str, compile_str):
        """find all matched"""
        return re.findall(re.compile(compile_str), find_str)

    @staticmethod
    def translate(translate_str, in_tab, out_tab):
        """字符串的翻译功能"""
        trantab = str.maketrans(in_tab, out_tab)  # 制作翻译表
        return translate_str.translate(trantab)

    @staticmethod
    def contain_zh(strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

if __name__ == "__main__":

    a = np.array([[1,2,3], [4,5,6], [7,8,9]])

    print(StrUtil.contain_zh("123456a好的sdd"))





















