# -*- coding: utf-8  -*-
# -*- author: jokker -*-

from JoTools.utils.FileOperationUtil import FileOperationUtil

img_dir = r"/home/ldq/input_dir/JPEGImages"
save_txt_path = r"./test_img_path.txt"

with open(save_txt_path, 'w') as txt_file:

    for each_img_path in FileOperationUtil.re_all_file(img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG']):
            txt_file.write(each_img_path)
            txt_file.write('\n')








