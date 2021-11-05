# -*- coding: utf-8  -*-
# -*- author: jokker -*-

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

# version 1.3.3.1 【核对完毕】


@DecoratorUtil.time_this
def dete_jyzZB(model_dict, data):

    try:
        model_jyzZB_1 = model_dict["model_jyzZB_1"]
        model_jyzZB_2 = model_dict["model_jyzZB_2"]

        # jyzZB step_1
        dete_res_jyzZB = model_jyzZB_1.detectSOUT(path=data['path'], image=copy.deepcopy(data['im']), image_name=data['name'])

        # jyzZB step_2
        result_res = DeteRes(assign_img_path=data['path'])
        result_res.img_path = data['path']
        dete_res_jyzZB.img_ndarry = data['im']
        #
        result_res.file_name = data['name']
        for each_dete_obj in dete_res_jyzZB:
            each_dete_obj.do_augment([150, 150, 150, 150], dete_res_jyzZB.width, dete_res_jyzZB.height, is_relative=False)
            # each_im = dete_res_jyzZB.get_sub_img_by_dete_obj(each_dete_obj)
            each_im = dete_res_jyzZB.get_sub_img_by_dete_obj_new(each_dete_obj)
            new_dete_res = model_jyzZB_2.detectSOUT(image=each_im, image_name=each_dete_obj.get_name_str())
            new_dete_res.offset(each_dete_obj.x1, each_dete_obj.y1)
            result_res += new_dete_res

        # new logic
        result_res.filter_tag1_by_tag2_with_nms(['jyzSingle'], ['jyzhead'], 0.5)
        result_res.do_nms_center_point(ignore_tag=True)
        result_res.update_tags({"jyzSingle": "jyzzb"})
        # torch.cuda.empty_cache()
        return result_res
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)

