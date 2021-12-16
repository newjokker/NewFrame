# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import os
import imagehash
from PIL import Image
from ..utils.HashlibUtil import HashLibUtil
from ..utils.FileOperationUtil import FileOperationUtil
from ..utils.PickleUtil import PickleUtil
from ..utils.DecoratorUtil import DecoratorUtil
import progressbar


class HashImageUtil(object):

    def __init__(self):
        self.db_dir_list = []           # 作为数据库的文件夹，每次扫描里面的文件，丰富数据库
        self.db_save_dir = None         # 用于保存本地缓存文件的路径，文件存储了文件名和文件 MD5 img_hash 等信息
        self.md5_hash_dict = {}         # 存储文件 md5 值和对应图片的 hash 值
        self.md5_file_name_dict = {}    # 存储文件 md5 值和文件名的字典
        self.hash_size = 20             # 计算图像 img_hash 所用的 size
        self.save_num = 500             # 每扫描多少张新图片就保存数据一次

    def update_db(self):
        """更新数据库，扫描指定的文件夹里面的所有，jpg，png 后缀的文件，并更新到数据库文件中"""
        index = 0
        for each_img_dir in self.db_dir_list:
            # 去除不是文件夹的路径
            if not os.path.isdir(each_img_dir):
                continue
            # 遍历指定文件夹下面的所有文件
            print("* 更新文件夹 : {0}".format(each_img_dir))
            file_img_list = FileOperationUtil.re_all_file(each_img_dir, lambda x:str(x).endswith((".jpg", ".png", ".JPG", ".PNG")))
            pb = progressbar.ProgressBar(len(file_img_list)).start()
            for img_index, each_img_path in enumerate(file_img_list):
                # 计算图片的 md5 值
                pb.update(img_index+1)
                # 每 500 张图片保存一下数据库
                if index % 500 == 0:
                    index += 1
                    # print("* 更新数据库文件")
                    self.save_dict_to_pkl()
                try:
                    each_file_md5 = HashLibUtil.get_file_md5(each_img_path)
                    # 是个新文件，没有计算过 img_hash
                    if not each_file_md5 in self.md5_hash_dict:
                        index += 1
                        # print(index, each_img_path)
                        self.md5_file_name_dict[each_file_md5] = each_img_path
                        # 计算图像的 hash 值
                        each_img_hash = imagehash.average_hash(Image.open(each_img_path), hash_size=self.hash_size)
                        self.md5_hash_dict[each_file_md5] = each_img_hash
                except Exception as e:
                    print(e)
            pb.finish()
        # 更新本地文件
        self.save_dict_to_pkl()

    def do_init(self):
        """初始化"""

        # 读取字典信息
        save_md5_hash_dict_path = os.path.join(self.db_save_dir, "{0}_{1}.pkl".format("md5_hash_dict", self.hash_size))
        save_md5_file_name_dict_path = os.path.join(self.db_save_dir, "{0}_{1}.pkl".format("md5_file_name_dict", self.hash_size))

        if os.path.isfile(save_md5_hash_dict_path):
            self.md5_file_name_dict = PickleUtil.load_data_from_pickle_file(save_md5_file_name_dict_path)

        if os.path.isfile(save_md5_file_name_dict_path):
            self.md5_hash_dict = PickleUtil.load_data_from_pickle_file(save_md5_hash_dict_path)

    def save_dict_to_pkl(self):
        """将两个重要的字典保存为本地文件"""
        # 文件名中要体现计算 hash 值用的 hash 位数
        save_md5_hash_dict_path = os.path.join(self.db_save_dir, "{0}_{1}.pkl".format("md5_hash_dict", self.hash_size))
        PickleUtil.save_data_to_pickle_file(self.md5_hash_dict, save_md5_hash_dict_path)
        save_md5_file_name_dict_path = os.path.join(self.db_save_dir, "{0}_{1}.pkl".format("md5_file_name_dict", self.hash_size))
        PickleUtil.save_data_to_pickle_file(self.md5_file_name_dict, save_md5_file_name_dict_path)

    def check_db(self):
        """查看数据库中是否有文件不存在，删除这些文件"""
        pass

    @DecoratorUtil.time_this
    def find_most_similar_img(self, assign_img_path, img_count=3):
        """找到最相似的几个图片"""
        # 计算图片的 img_hash
        img_hash = imagehash.average_hash(Image.open(assign_img_path), hash_size=self.hash_size)
        # 对比图片的 img_hash 和库中所有的 img_hash
        find_res = []
        for each_hash_info in self.md5_hash_dict.items():
            find_res.append((each_hash_info[1] - img_hash, each_hash_info[0]))
        # 结果进行排序，返回前 img_count 个结果
        find_res = sorted(find_res, key=lambda x:x[0], reverse=False)
        # 打印前 n 个最相似的结果
        print("当前结果，扫描对比 {0} 张图片".format(len(find_res)))
        res = []
        for each in find_res[:img_count]:
            # print(each[0], self.md5_file_name_dict[each[1]])
            res.append((each[0], self.md5_file_name_dict[each[1]]))
        return res

    def do_process(self):
        """主流程"""
        self.do_init()
        # self.update_db()


if __name__ == "__main__":


    a = HashImageUtil()
    # 指定缓存文件存放文件夹
    a.db_save_dir = r"C:\Users\14271\Desktop\del\db_temp"
    # 扫描到多少张新图片保存一次数据库
    a.save_num = 1000
    # 指定数据库文件夹
    a.db_dir_list = [
        r"C:\data\fzc_优化相关资料\dataset_fzc\001_图片大小转为1280\train_img",
        r"C:\data\fzc_优化相关资料\dataset_fzc\001_图片大小转为1280\res_img_dir",]
    # 初始化
    a.do_init()
    a.update_db()
    # 需要数据库中指定相似的图片
    res = a.find_most_similar_img(r"C:\Users\14271\Desktop\save_res_2\test.jpg", img_count=1)

    for each in res:
        print(each)