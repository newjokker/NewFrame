# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import os
import time
import requests
from multiprocessing import Pool
from JoTools.utils.FileOperationUtil import FileOperationUtil
from JoTools.txkjRes.deteRes import DeteRes


def post_img(each_img_path):
    global post_img_num
    post_img_num += 1
    url = r"http://192.168.3.74:3232/receive_server/post_img"
    res = requests.post(url=url, data={'filename': os.path.split(each_img_path)[1]}, files={'image': open(each_img_path, 'rb')})
    print("{0} : {1}".format(post_img_num, res.text.strip()))


if __name__ == "__main__":

    start_time = time.time()

    post_img_num = 0
    img_dir = r"/home/ldq/fangtian_test/fangtian_nc_kkx"
    img_path_list = list(FileOperationUtil.re_all_file(img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG']))

    pool = Pool()
    pool.map(post_img, img_path_list)
    pool.close()
    pool.join()

    print("use time : {0}".format(time.time() - start_time))