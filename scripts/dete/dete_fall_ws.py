# -*- coding: utf-8  -*-
# -*- author: jokker -*-

# 跌倒检测

from lib.JoTools.txkjRes.resTools import ResTools
from lib.JoTools.utils.FileOperationUtil import FileOperationUtil
from lib.JoTools.utils.CsvUtil import CsvUtil
from lib.JoTools.txkjRes.deteRes import DeteRes
from lib.JoTools.txkjRes.deteObj import DeteObj
from lib.JoTools.txkjRes.deteAngleObj import DeteAngleObj
from lib.JoTools.utils.JsonUtil import JsonUtil
#
from JoTools.utils.DecoratorUtil import DecoratorUtil
import copy


@DecoratorUtil.time_this
def dete_fall_ws(model_dict, data):
    try:
        model_pose = model_dict["model_fall_ws_pose"]
        model_human = model_dict["model_fall_ws_human"]

        #
        human_dete_res = model_human.detectSOUT(path=data['path'], image=copy.deepcopy(data['im']),
                                                image_name=data['name'])
        # 转换出 boxes 用以检测关键点
        boxes = []
        for each_dete_obj in human_dete_res:
            boxes.append([each_dete_obj.x1, each_dete_obj.y1, each_dete_obj.x2, each_dete_obj.y2])

        point_res = model_pose.detectSOUT(copy.deepcopy(data['im']), np.array(boxes))

        human_dete_res.print_as_fzc_format()
        print("-" * 50)
        point_res.print_as_fzc_format()
        print("-" * 50)

        save_path = os.path.join(r"/home/ldq/NewFrame/res", data['name'])
        print(save_path)
        point_res.img_path = data['path']
        point_res.draw_res(save_path)

        return human_dete_res

    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)






