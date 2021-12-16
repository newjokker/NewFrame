# -*- coding: utf-8  -*-
# -*- author: jokker -*-

# 参考：https://www.liaoxuefeng.com/wiki/1016959663602400/1017686752491744

"""
* Python的hashlib提供了常见的摘要算法，如MD5，SHA1等等。
* 什么是摘要算法呢？摘要算法又称哈希算法、散列算法。它通过一个函数，把任意长度的数据转换为一个长度固定的数据串（通常用16进制的字符串表示）。

* 摘要算法就是通过摘要函数f()对任意长度的数据data计算出固定长度的摘要digest，目的是为了发现原始数据是否被人篡改过。

* 摘要算法之所以能指出数据是否被篡改过，就是因为摘要函数是一个单向函数，计算f(data)很容易，但通过digest反推data却非常困难。
而且，对原始数据做一个bit的修改，都会导致计算出的摘要完全不同。
"""

import hashlib
import os
import shutil
from ..utils.FileOperationUtil import FileOperationUtil
from ..utils.PickleUtil import PickleUtil

"""
# # 如果数据量很大，可以分块多次调用update()，最后计算的结果是一样的：
"""


class HashLibUtil(object):

    @staticmethod
    def get_file_md5(file_path):
        """获取文件的 MD5 值"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as xml_file:
            md5.update(xml_file.read())
            return md5.hexdigest()

    @staticmethod
    def is_the_same_file(file_path_1, file_path_2):
        """判断两个文件是否是一个文件"""
        md5_1 = HashLibUtil.get_file_md5(file_path_1)
        md5_2 = HashLibUtil.get_file_md5(file_path_2)
        return True if md5_1 == md5_2 else False

    @staticmethod
    def duplicate_checking(file_path_list):
        """文件查重，输出重复的文件，放在一个列表里面
        DS : [[file_path_1, file_path_2], []]"""
        file_md5 = {}
        res = []
        # 计算所有文件的 md5 值
        for index, each_file_path in enumerate(file_path_list):
            print(index, each_file_path)
            md5 = HashLibUtil.get_file_md5(each_file_path)
            if md5 in file_md5:
                file_md5[md5].append(each_file_path)
            else:
                file_md5[md5] = [each_file_path]
        # 将有重复的文件放到列表中
        for each_md5 in file_md5:
            if len(file_md5[each_md5]) > 1:
                res.append(file_md5[each_md5])
        return res

    @staticmethod
    def leave_one(img_dir, save_dir=None, endswith=(".jpg", ".png", ".JPG", ".PNG"), del_log_path=None):
        """检查路径下面有没有重复的文件，把所有不重复的文件复制到指定文件夹，或者直接在当前文件夹删除"""
        md5_set = set()
        del_list = []
        file_count_sum = 0
        del_file_count = 0
        for each_img_path in FileOperationUtil.re_all_file(img_dir, lambda x:str(x).endswith(endswith)):
            each_md5 = HashLibUtil.get_file_md5(each_img_path)
            file_count_sum += 1
            if each_md5 not in md5_set:
                md5_set.add(each_md5)
                if save_dir is not None:
                    each_save_path = os.path.join(save_dir, os.path.split(each_img_path)[1])
                    shutil.copyfile(each_img_path, each_save_path)
            else:
                if save_dir is None:
                    # 不另存为文件夹，就直接在当前文件夹中将重复的图片删除
                    print("remove : {0}".format(each_img_path))
                    del_list.append((each_md5, each_img_path))
                    os.remove(each_img_path)
                    del_file_count += 1
        # 保存删除的 log
        if del_log_path:
            with open(del_log_path, 'a') as log_txt:
                for each_line in del_list:
                    log_txt.write(each_line[0] + ' : ' + each_line[1])
                    log_txt.write('\n')
        return file_count_sum, del_file_count

    @staticmethod
    def save_file_md5_to_pkl(file_dir, save_pkl_path, need_file_type=None, assign_file_path_file=None, each_file_count=1000):
        """将制定路径下面的所有文件的 md5 和 路径组成的字典保存到 pkl 文件中"""

        # 执行需要计算 md5 值的数据的类型
        if need_file_type is None:
            need_file_type = ['.jpg', '.JPG', '.png', '.PNG']
        # 可以指定过滤已经扫描过的目录
        if assign_file_path_file:
            md5_dict = PickleUtil.load_data_from_pickle_file(assign_file_path_file)
            file_path_set = set()
            for each_file_Path_set in md5_dict.values():
                file_path_set = set.union(each_file_Path_set, file_path_set)
        else:
            file_path_set = set()
            md5_dict = {}

        print('-'*50)
        print("start file_path_set length   : ", len(file_path_set))
        print("start md5 dict length        : ", len(md5_dict))

        try:
            find_index = 0
            for each_file_path in FileOperationUtil.re_all_file(file_dir, endswitch=need_file_type):
                # 过滤已经扫描过的目录
                if each_file_path not in file_path_set:
                    file_path_set.add(each_file_path)
                else:
                    continue

                find_index += 1
                print(find_index, each_file_path)

                # save file
                if find_index % each_file_count == 0:
                    PickleUtil.save_data_to_pickle_file(md5_dict, save_pkl_path)

                each_md5 = HashLibUtil.get_file_md5(each_file_path)

                if each_md5 not in md5_dict:
                    md5_dict[each_md5] = set()
                    md5_dict[each_md5].add(each_file_path)
                else:
                    md5_dict[each_md5].add(each_file_path)

        except Exception as e:
            print('GOT ERROR---->')
            print(e)
            print(e.__traceback__.tb_frame.f_globals["__file__"])
            print(e.__traceback__.tb_lineno)

        finally:
            # save_to_pickle
            PickleUtil.save_data_to_pickle_file(md5_dict, save_pkl_path)
            print("stop file_path_set length    : ", len(file_path_set))
            print("stop md5 dict length         : ", len(md5_dict))

        return md5_dict

if __name__ == "__main__":

    # md5_str = HashLibUtil.get_file_md5(r"C:\Users\14271\Desktop\face_detection\human_face\cy\affd45ad896ddffaa1a8fdf95c5d87d0.jpg")
    # print(md5_str)
    #
    # exit()

    file_list = []
    file_list.append(list(FileOperationUtil.re_all_file(r"C:\data\fzc_优化相关资料\000_等待训练", lambda x:str(x).endswith(('.jpg', '.JPG')))))

    a = HashLibUtil.duplicate_checking(file_list)

    for each in a:
        print(each)


    # filePath1 = r'C:\Users\Administrator\Desktop\for6.xml'
    # filePath2 = r'C:\Users\Administrator\Desktop\for6 - 副本.xml'
    #
    # a = HashLibUtil.is_the_same_file(filePath1, filePath2)

    # file_dir = r'C:\Users\14271\Desktop\del\深度学习\effi\food_challenge2\del'
    # file_list_1 = FileOperationUtil.re_all_file(r"C:\data\test_data\fzc\原始faster训练用的图片fzc\origin", lambda x:str(x).endswith('.jpg'))
    # file_list_2 = FileOperationUtil.re_all_file(r"C:\Users\14271\Desktop\防振锤优化\000_标准测试集\img", lambda x:str(x).endswith('.jpg'))
    # file_list = file_list_1 + file_list_2
    # dc = HashLibUtil.duplicate_checking(file_list)
    #
    # for each in dc:
    #     print(each)
    #     print("-"*100)

    HashLibUtil.leave_one(r"C:\Users\14271\Desktop\fzc报错的地方",
                          r"C:\Users\14271\Desktop\报错文件去重")



    # # fixme 删除第一个文件外的其他文件
    # import os
    # for each in dc:
    #     for i in each[1:]:
    #         print(i)
    #         os.remove(i)

