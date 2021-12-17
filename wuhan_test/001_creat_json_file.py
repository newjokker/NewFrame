# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
from JoTools.utils.JsonUtil import JsonUtil
from JoTools.utils.FileOperationUtil import FileOperationUtil

img_dir = r"/home/ldq/input_dir"
json_path = r"./pictureName.json"
json_info = []

for each_img_path in FileOperationUtil.re_all_file(img_dir, endswitch=['.jpg', '.JPG', '.png', '.PNG']):

    json_info.append({
        "fileName": os.path.split(each_img_path)[1],
        "originFileName": "originFileName_" + os.path.split(each_img_path)[1]
    })

JsonUtil.save_data_to_json_file(json_info, json_path)









