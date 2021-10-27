# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import os
import requests
from multiprocessing import Pool
from JoTools.utils.FileOperationUtil import FileOperationUtil
from JoTools.txkjRes.deteRes import DeteRes


def post_img(each_img_path):
    global post_img_num
    post_img_num += 1
    res = requests.post(url=url, data={'filename': os.path.split(each_img_path)[1]}, files={'image': open(each_img_path, 'rb')})
    print("{0} : {1}".format(post_img_num, res))


if __name__ == "__main__":


    post_img_num = 0

    img_dir = r""

    img_path_list = list(FileOperationUtil.re_all_file(img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG']))

    pool = Pool(10)
    pool.map(post_img, img_path_list)
    pool.close()
    pool.join()
