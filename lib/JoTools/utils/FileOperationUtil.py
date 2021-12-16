# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import shutil
import os
import datetime
import collections


class FileOperationUtil(object):
    """文件操作类"""

    @staticmethod
    def delete_folder(dir_path):
        """删除一个路径下的所有文件"""
        # todo 这个一般都是最后一步，所以如何解决删除文件报错的问题，可能是文件被程序在占用
        shutil.rmtree(dir_path)

    @staticmethod
    def create_folder(folder_path):
        """如果文件夹不存在创建文件夹，如果文件夹存在选择是否清空文件夹"""
        os.makedirs(folder_path, exist_ok=True)

    @staticmethod
    def bang_path(file_path):
        """将文件名给bang开，这样省的麻烦，只能是文件地址，文件夹不好 bang 开"""
        if not os.path.isfile(file_path):
            raise EOFError ("need correct file path")

        folder_path = os.path.split(file_path)
        file_name = os.path.splitext(folder_path[1])[0]
        file_suffix = os.path.splitext(folder_path[1])[1]
        return folder_path[0], file_name, file_suffix

    @staticmethod
    def re_all_file(file_path, func=None, endswitch=None):
        """返回文件夹路径下的所有文件路径（搜索文件夹中的文件夹）"""

        if not os.path.isdir(file_path):
            print("* not folder path")
            raise EOFError

        # result = []
        for i, j, k in os.walk(file_path):
            for each_file_name in k:

                # 过滤后缀不符合的路径
                if endswitch is not None:
                    _, end_str = os.path.splitext(each_file_name)
                    if end_str not in endswitch:
                        continue

                abs_path = i + os.sep + each_file_name
                if func is None:
                    # result.append(abs_path)
                    yield abs_path
                else:
                    if func(abs_path):
                        # result.append(os.path.join(i, each_file_name))
                        yield os.path.join(i, each_file_name)
        # return result

    @staticmethod
    def re_all_folder(folder_path):
        """返回找到的所有文件夹的路径"""
        # fixme 可以用生成器写，这样就不用等扫描完了再去操作了
        if not os.path.isdir(folder_path):
            print(" 不是文件夹路径 ")
            raise EOFError

        # result = []
        for i, j, k in os.walk(folder_path):
            for each_dir_name in j:
                abs_path = i + os.sep + each_dir_name
                # result.append(abs_path)
                yield abs_path
        # return result

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
            #
            new_file_dir = os.path.dirname(new_file_path)
            if not os.path.exists(new_file_dir):
                os.makedirs(new_file_dir)
            #
            if is_clicp:
                shutil.move(each_file_path, new_file_path)
            else:
                shutil.copyfile(each_file_path, new_file_path)

    @staticmethod
    def merge_root_dir(root_dir_1, root_dir_2, is_clip=False):
        """对 root 文件夹进行合并，两个 root 文件夹及其包含的子文件夹，A(a,b,c), B(b,c,d) 那么将 A B 中的 b,c 文件夹中的内容进行合并，并复制 a, d , is_clip 是否使用剪切的方式进行合并"""
        for each_name in os.listdir(root_dir_2):
            each_path = os.path.join(root_dir_2, each_name)
            each_releate_path = os.path.join(root_dir_1, each_name)

            if os.path.isdir(each_path):
                if os.path.exists(each_releate_path):
                    FileOperationUtil.merge_root_dir(each_releate_path, each_path)
                else:
                    shutil.move(each_path, each_releate_path)   # 递归
            else:
                if os.path.exists(each_releate_path):
                    print("* {0} has exists".format(each_releate_path))
                else:
                    if is_clip is False:
                        shutil.copy(each_path, each_releate_path)
                    else:
                        shutil.move(each_path, each_releate_path)

    # ------------------------------------ need repair -----------------------------------------------------------------

    @staticmethod
    def get_father_path(str_temp):
        """ 查找父文件夹，mac 和 windows 环境下都能运行"""
        # 去掉末尾的 '\' 和 '/'
        # str_temp = str_temp.rstrip(r'/')
        # str_temp = str_temp.rstrip(r'\\')
        str_temp = str_temp.rstrip(os.sep)
        return os.path.split(str_temp)[0]

    @staticmethod
    def clear_empty_folder(dir_path):
        """删除空文件夹, 考虑文件夹中只有空文件夹的情况，出现的话需要再次跑一遍程序，遍历删除文件夹"""
        del_num = 0
        for each_folder_path in FileOperationUtil.re_all_folder(dir_path):
            if not os.listdir(each_folder_path):
                shutil.rmtree(each_folder_path)
                del_num += 1
                print("* del {0}".format(each_folder_path))
                if del_num >= 1:
                    FileOperationUtil.clear_empty_folder(dir_path)

    @staticmethod
    def find_diff_file(file_list_1, file_list_2, func=None):
        """根据文件名是否相同定义文件是否相同，对比两个列表中的文件的差异"""
        file_name_dict_1, file_name_dict_2 = dict(), dict()
        #
        for each_file_path in file_list_1:
            each_file_name = os.path.split(each_file_path)[1]
            each_file_name = func(each_file_name) if func is not None else each_file_name
            file_name_dict_1[each_file_name] = each_file_path
        #
        for each_file_path in file_list_2:
            each_file_name = os.path.split(each_file_path)[1]
            each_file_name = func(each_file_name) if func is not None else each_file_name
            file_name_dict_2[each_file_name] = each_file_path

        # 那些 a 不在 b 中 , b 不在 a 中的，都记下来
        res = {"inanotb":[], "inbnota":[]}
        #
        for each_name_1 in file_name_dict_1:
            if each_name_1 not in file_name_dict_2:
                res["inanotb"].append(file_name_dict_1[each_name_1])
        #
        for each_name_2 in file_name_dict_2:
            if each_name_2 not in file_name_dict_1:
                res["inbnota"].append(file_name_dict_2[each_name_2])

        return res

    @staticmethod
    def divide_file_equally(file_dir, save_dir, divide_count=3, need_endswitch=None, assign_name='part_', is_clip=False):
        """均分文件"""

        # 初始化数据结构
        file_dict = {}
        for i in range(divide_count):
            file_dict[i] = []

        # 遍历分配数据
        index = 0
        for each_file in FileOperationUtil.re_all_file(file_dir, endswitch=need_endswitch):
            file_dict[index].append(each_file)
            if index < divide_count-1:
                index += 1
            else:
                index = 0
            print(index)

        # 移动数据
        for each_key in file_dict:
            each_save_dir = os.path.join(save_dir, assign_name+str(each_key+1))
            os.makedirs(each_save_dir, exist_ok=True)
            FileOperationUtil.move_file_to_folder(file_dict[each_key], assign_folder=each_save_dir, is_clicp=is_clip)

    @staticmethod
    def show_file_dispersed(file_dir, endswitch, assign_func):
        """查看文件的分布"""

        data_list = []

        for each_file in FileOperationUtil.re_all_file(file_dir, endswitch=endswitch):
            data_list.append(assign_func(each_file))
        # 统计
        a = collections.Counter(data_list)
        # 显示

        print(list(a.keys()))

        min_value = min(list(a.keys()))
        max_value = max(list(a.keys()))

        for i in range(min_value, max_value + 1):
            if i in a:
                print(i, a[i])

        print(a)



if __name__ == "__main__":

    #FileOperationUtil.clear_empty_folder(r"C:\Users\14271\Desktop\del")

    file_dir = r"/home/suanfa-1/ceshiji"

    FileOperationUtil.show_file_dispersed(file_dir, endswitch=['.jpg', '.JPG'], assign_func=lambda x:int(os.path.getsize(x)/(1024*1024)))


